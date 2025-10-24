// app/callback/page.tsx
'use client';

import { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';

export default function CallbackPage() {
    const searchParams = useSearchParams();
    const [logMessage, setLogMessage] = useState("인증 코드 수신 및 검증 중...");
    const router = useRouter();
    const OAUTH_STATE_KEY = 'oauth_state';
    
    useEffect(() => {
        const authCode = searchParams.get('code');
        const error = searchParams.get('error');
        const returnedState = searchParams.get('state');

        if (error) {
            setLogMessage(`❌ 인증 실패 오류: ${error}`);
            return;
        }

        if (authCode) {
            // State 값 검증 (CSRF 방지)
            const originalState = sessionStorage.getItem(OAUTH_STATE_KEY);
            sessionStorage.removeItem(OAUTH_STATE_KEY);

            if (returnedState !== originalState) {
                setLogMessage(`🚨 상태 검증 실패! CSRF 공격 가능성. 원래 State: ${originalState}`);
                // 보안을 위해 실패 시 로그인 페이지로 리다이렉션 권장
                // router.push('/login'); 
                return;
            }

            // --- 이 부분이 핵심입니다: Access Token 교환 ---
            setLogMessage(`✅ Google OAuth 인증 성공!
            
**다음 단계: 이 인증 코드(code)를 백엔드 API로 보내 Access Token을 교환해야 합니다.**

수신된 인증 코드 (Code):
${authCode}

<이후 여기에 백엔드 API 호출 로직을 구현합니다.>`);
            // ---------------------------------------------
        } else if (searchParams.toString().length > 0) {
             setLogMessage(`ℹ️ 예상치 못한 콜백입니다. URL 파라미터가 있지만 'code'가 없습니다.`);
        } else {
             setLogMessage("콜백 URL로 접근했지만, 'code' 파라미터가 없습니다.");
        }
    }, [searchParams, router]); 

    return (
        <div style={{ padding: 50 }}>
            <h1>인증 콜백 처리</h1>
            <pre style={{ background: '#f4f4f4', padding: '15px', borderRadius: '4px', whiteSpace: 'pre-wrap' }}>
                {logMessage}
            </pre>
            <button 
                onClick={() => router.push('/login')} 
                style={{ marginTop: '20px', padding: '10px' }}
            >
                다시 로그인 페이지로
            </button>
        </div>
    );
}