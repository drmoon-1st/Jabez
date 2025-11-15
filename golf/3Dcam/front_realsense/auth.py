"""
auth.py

간단한 Authorization Code + PKCE (loopback) 구현.

사용법:
  from auth import AuthClient
  ac = AuthClient.from_config('config.json')
  tokens = ac.login_and_get_tokens()

반환값: dict with keys 'access_token', 'id_token', 'refresh_token'(선택)
"""
import json
import os
from dotenv import load_dotenv
import base64
import hashlib
import secrets
import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
import requests
from pathlib import Path
import time


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode('ascii')


class _CallbackHandler(BaseHTTPRequestHandler):
    server_version = "FRAuth/0.1"
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        # store the query on server for retrieval
        self.server.last_query = qs
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write("<html><body><h2>인증 완료. 이 창을 닫으십시오.</h2></body></html>".encode('utf-8'))

    def log_message(self, format, *args):
        return


class AuthClient:
    def __init__(self, cognito_domain: str, client_id: str, redirect_uri: str, scope: str = 'openid profile'):
        self.cognito_domain = cognito_domain.rstrip('/')
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.scope = scope
        # optional backend base for mediated login
        self.backend_base = os.getenv('BACKEND_BASE') or None

    @classmethod
    def from_config(cls, path: str = 'config.json'):
        # Try JSON config first, then fallback to environment (.env)
        p = Path(path)
        if p.exists():
            cfg = json.loads(p.read_text(encoding='utf-8'))
            return cls(cfg['COGNITO_DOMAIN'], cfg['CLIENT_ID'], cfg['REDIRECT_URI'], cfg.get('SCOPE', 'openid profile'))

        # Fallback: load from environment (.env)
        load_dotenv()  # load .env into environment if present
        cognito = os.getenv('COGNITO_DOMAIN') or os.getenv('COGNITO_DOMAIN_URL')
        client_id = os.getenv('CLIENT_ID')
        redirect = os.getenv('REDIRECT_URI')
        scope = os.getenv('SCOPE', 'openid profile')
        if not (cognito and client_id and redirect):
            raise FileNotFoundError(f"config.json not found and required env vars missing: COGNITO_DOMAIN/CLIENT_ID/REDIRECT_URI")
        return cls(cognito, client_id, redirect, scope)

    def _start_local_server(self, host='127.0.0.1', port=5000, timeout=120):
        server = HTTPServer((host, port), _CallbackHandler)
        server.last_query = None

        def _serve():
            try:
                server.timeout = timeout
                server.handle_request()
            except Exception:
                pass

        t = threading.Thread(target=_serve, daemon=True)
        t.start()
        return server

    def login_and_get_tokens(self, open_browser: bool = True, timeout: int = 120) -> dict:
        # If backend_base is set, use backend-mediated login flow
        if self.backend_base:
            # call backend to create session and get auth_url
            start_url = f"{self.backend_base}/api/auth/start"
            try:
                r = requests.post(start_url, timeout=5)
                r.raise_for_status()
                payload = r.json()
                session_id = payload.get('session_id')
                auth_url = payload.get('auth_url')
            except Exception as e:
                raise RuntimeError(f'Failed to start backend auth session: {e}')

            if open_browser:
                webbrowser.open(auth_url)
            else:
                print('Open this URL in a browser:', auth_url)

            # poll backend for result
            result_url = f"{self.backend_base}/api/auth/result"
            waited = 0
            poll_interval = 1.0
            while waited < timeout:
                try:
                    pr = requests.get(result_url, params={'session_id': session_id}, timeout=5)
                    if pr.status_code == 200:
                        try:
                            tokens = pr.json()
                        except Exception as e:
                            raise RuntimeError(f'Failed to parse tokens from backend: {e}')
                        return tokens
                    elif pr.status_code == 202:
                        # pending
                        pass
                    else:
                        # 404 or other
                        raise RuntimeError(f'Auth result error: status={pr.status_code} body={pr.text}')
                except requests.exceptions.RequestException:
                    pass
                time.sleep(poll_interval)
                waited += poll_interval
            raise TimeoutError('Timed out waiting for backend auth result')

        # Fallback: original PKCE local loopback flow
        # generate PKCE code_verifier and code_challenge
        code_verifier = _b64url_encode(secrets.token_bytes(32))
        code_challenge = _b64url_encode(hashlib.sha256(code_verifier.encode('utf-8')).digest())

        # start local listener
        parsed = urllib.parse.urlparse(self.redirect_uri)
        host = parsed.hostname or '127.0.0.1'
        port = parsed.port or 5000
        server = self._start_local_server(host=host, port=port, timeout=timeout)

        # build auth url
        auth_params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': self.scope,
            'code_challenge_method': 'S256',
            'code_challenge': code_challenge,
            'state': secrets.token_urlsafe(8)
        }
        auth_url = f"{self.cognito_domain}/oauth2/authorize?{urllib.parse.urlencode(auth_params)}"
        if open_browser:
            webbrowser.open(auth_url)
        else:
            print('Open this URL in a browser:', auth_url)

        # wait for callback (single request)
        waited = 0
        while server.last_query is None and waited < timeout:
            time.sleep(0.5)
            waited += 0.5

        if not server.last_query:
            raise TimeoutError('Auth callback not received')

        qs = server.last_query
        if 'error' in qs:
            raise RuntimeError('Auth error: ' + str(qs.get('error')))

        code = qs.get('code', [None])[0]
        if not code:
            raise RuntimeError('Auth code not found in callback')

        # Exchange code for tokens using PKCE (no client_secret)
        token_url = f"{self.cognito_domain}/oauth2/token"
        data = {
            'grant_type': 'authorization_code',
            'client_id': self.client_id,
            'code': code,
            'redirect_uri': self.redirect_uri,
            'code_verifier': code_verifier
        }
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        # Debug: show what we're sending (no secrets included)
        print('DEBUG: Exchanging code for tokens')
        print('DEBUG: token_url=', token_url)
        print('DEBUG: client_id=', self.client_id)
        print('DEBUG: redirect_uri=', self.redirect_uri)
        print('DEBUG: code=', code)
        print('DEBUG: code_verifier_len=', len(code_verifier) if code_verifier else None)

        r = requests.post(token_url, data=data, headers=headers, timeout=10)
        # Provide more helpful error output when Cognito returns 4xx/5xx
        if r.status_code != 200:
            # include response body for diagnosis
            body = r.text
            raise RuntimeError(f'Token exchange failed: status={r.status_code} url={token_url} response={body}')
        try:
            tokens = r.json()
        except Exception as e:
            raise RuntimeError(f'Failed to parse token JSON response: {e}; body={r.text}')
        # expected keys: access_token, id_token, refresh_token
        return tokens


if __name__ == '__main__':
    # quick test
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.json')
    args = p.parse_args()
    client = AuthClient.from_config(args.config)
    print('Opening browser for login...')
    t = client.login_and_get_tokens()
    print('Tokens:', list(t.keys()))
