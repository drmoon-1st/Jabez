// app/login/page.tsx
'use client';

import { useRouter } from 'next/navigation';

export default function LoginPage() {
  const router = useRouter();

  const handleLogin = () => {
    // CSRF 방지를 위한 state 값 생성 및 저장
    const state = Math.random().toString(36).substring(2, 15);
    sessionStorage.setItem('oauth_state', state);

    // 환경 변수 로드
    const COGNITO_DOMAIN = process.env.NEXT_PUBLIC_COGNITO_DOMAIN;
    const CLIENT_ID = process.env.NEXT_PUBLIC_CLIENT_ID;
    const REDIRECT_URI = process.env.NEXT_PUBLIC_REDIRECT_URI;
    const SCOPE = process.env.NEXT_PUBLIC_SCOPE;
    const RESPONSE_TYPE = process.env.NEXT_PUBLIC_RESPONSE_TYPE;

    if (!COGNITO_DOMAIN || !CLIENT_ID || !REDIRECT_URI) {
        alert("🚨 환경 변수 설정이 누락되었습니다. .env.local 파일을 확인하세요.");
        return;
    }

    // Hosted UI로 리다이렉션할 URL 구성
    const authUrl = 
      `${COGNITO_DOMAIN}/oauth2/authorize?` +
      `response_type=${RESPONSE_TYPE}&` +
      `client_id=${CLIENT_ID}&` +
      `redirect_uri=${REDIRECT_URI}&` +
      `scope=${SCOPE}&` +
      `state=${state}`;
      
    // 브라우저 리다이렉션
    router.push(authUrl);
  };

  return (
    <div style={{ padding: 50, textAlign: 'center', color: 'var(--foreground)' }}>
      <h1>Cognito 로그인 테스트</h1>
      <button
        onClick={handleLogin}
        className="btn-primary"
        style={{ fontSize: '16px', marginTop: '20px' }}
      >
        Google 로그인 시작 (Hosted UI로 이동)
      </button>
      <p style={{ marginTop: '20px', color: 'gray' }}>
        클릭 시, 등록된 콜백 주소({process.env.NEXT_PUBLIC_REDIRECT_URI})로 돌아옵니다.
      </p>
    </div>
  );
}