// app/callback/page.tsx
'use client';

import { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';

// FastAPI 백엔드 API 주소 정의
// .env 파일에서 불러오는 것이 권장되지만, 여기서는 명시적으로 지정합니다.
const BACKEND_TOKEN_API = process.env.NEXT_PUBLIC_API_BASE_URL + "/api/token";
const OAUTH_STATE_KEY = 'oauth_state';

export default function CallbackPage() {
    const searchParams = useSearchParams();
    const [logMessage, setLogMessage] = useState("인증 코드 수신 및 검증 중...");
    const [tokenData, setTokenData] = useState<any | null>(null); // 토큰 데이터를 저장할 상태
    const router = useRouter();

    useEffect(() => {
        const authCode = searchParams.get('code');
        const error = searchParams.get('error');
        const returnedState = searchParams.get('state');

        if (error) {
            setLogMessage(`❌ 인증 실패 오류: ${searchParams.get('error_description') || error}`);
            return;
        }

        const exchangeToken = async (code: string) => {
            if (!BACKEND_TOKEN_API) {   // typescript에서 필요한 null 체크, 안하면 fetch시 경고 날림
                setLogMessage("❌ 백엔드 API URL이 설정되어 있지 않습니다. NEXT_PUBLIC_BACKEND_API 환경변수를 확인하고 서버를 재시작하세요.");
                return;
            }

            setLogMessage(`✅ Google OAuth 인증 성공!
백엔드 API (${BACKEND_TOKEN_API})로 토큰 교환 요청 중... (HttpOnly 쿠키 설정 예정)`);
            
            try {
                // FastAPI의 Form(...) 인자를 위해 FormData를 사용합니다.
                const formData = new FormData();
                formData.append('code', code);
                
                const response = await fetch(BACKEND_TOKEN_API, {
                    method: 'POST',
                    body: formData, 
                    // ⭐ 중요: 백엔드가 설정한 쿠키를 수신하기 위해 credentials 포함 필요
                    credentials: 'include', 
                });

                if (response.ok) {
                    // 1. 성공! 브라우저가 Set-Cookie 헤더를 통해 쿠키를 이미 저장했습니다.
                    const data = await response.json(); // 성공 메시지 본문 읽기
                    
                    setLogMessage(`🎉 토큰 교환 성공!
                    ${data.message || '쿠키가 안전하게 설정되었습니다.'}`);
                    
                    // 2. 💡 문제 해결: router.push() 대신 router.replace() 사용
                    //    code 파라미터가 포함된 현재 URL을 '/uploader'로 대체하여 루프를 방지합니다.
                    router.replace('/uploader'); // <--- 이 부분을 push에서 replace로 변경
                    
                } else {
                    // 3. 토큰 교환 실패
                    const data = await response.json();
                    setLogMessage(`❌ 토큰 교환 실패: 
FastAPI 응답 오류: ${data.detail?.message || data.message || '알 수 없는 오류'}`);
                }
            } catch (e) {
                // 네트워크 오류 또는 서버 연결 실패
                setLogMessage(`❌ 통신 오류: FastAPI 서버에 연결할 수 없습니다. 서버(29001)가 실행 중인지 확인하세요.
오류 상세: ${e instanceof Error ? e.message : String(e)}`);
            }
        };

        if (authCode) {
            // State 값 검증 (CSRF 방지)
            const originalState = sessionStorage.getItem(OAUTH_STATE_KEY);
            sessionStorage.removeItem(OAUTH_STATE_KEY);

            if (returnedState !== originalState || !originalState) {
                setLogMessage(`🚨 상태 검증 실패! CSRF 공격 가능성.
돌아온 State: ${returnedState} / 원래 State: ${originalState}`);
                // 보안을 위해 실패 시 처리가 필요합니다.
                return;
            }
            
            // State 검증 성공 후 토큰 교환 시작
            exchangeToken(authCode);

        } else if (searchParams.toString().length > 0) {
            setLogMessage(`ℹ️ 예상치 못한 콜백입니다. URL 파라미터가 있지만 'code'가 없습니다.`);
        } else {
            setLogMessage("콜백 URL로 접근했지만, 'code' 파라미터가 없습니다.");
        }
    }, [searchParams, router]); 

    return (
        <div style={{ padding: 50, color: 'var(--foreground)' }}>
            <h1>인증 콜백 처리</h1>
            <pre className="card-muted" style={{ whiteSpace: 'pre-wrap' }}>
                {logMessage}
            </pre>

            {tokenData && (
                <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '15px', backgroundColor: '#f9f9f9' }}>
                    <h3>수신된 토큰 정보 (ID Token을 안전하게 저장해야 합니다):</h3>
                    <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word', fontSize: '12px' }}>
                        {JSON.stringify(tokenData, null, 2)}
                    </pre>
                </div>
            )}

            <button
                onClick={() => router.push('/login')}
                className="btn-primary"
                style={{ marginTop: '20px' }}
            >
                다시 로그인 페이지로
            </button>
        </div>
    );
}