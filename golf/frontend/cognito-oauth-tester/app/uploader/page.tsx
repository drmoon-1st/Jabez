// app/uploader/page.tsx
'use client';

import { useRouter } from 'next/navigation';
import { useState, useRef } from 'react';

// ⭐ 환경 변수에서 베이스 URL을 가져와 /upload 엔드포인트를 구성합니다.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;
// 변경: 백엔드의 실제 엔드포인트를 /api/upload 로 맞춥니다.
const BACKEND_UPLOAD_API = `${API_BASE_URL}/api/upload/`;

export default function UploaderPage() {
    const router = useRouter();
    const [file, setFile] = useState<File | null>(null);
    const [uploadStatus, setUploadStatus] = useState<string>('업로드할 파일을 선택해주세요.');
    const fileInputRef = useRef<HTMLInputElement>(null);

    // ----------------------------------------------------
    // 1. 파일 선택 핸들러
    // ----------------------------------------------------
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            const selectedFile = e.target.files[0];
            setFile(selectedFile);
            setUploadStatus(`파일 선택 완료: ${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`);
        } else {
            setFile(null);
        }
    };

    // ----------------------------------------------------
    // 2. 업로드 프로세스 시작
    // ----------------------------------------------------
    const handleUpload = async () => {
        if (!file) {
            alert('먼저 파일을 선택해주세요.');
            return;
        }

        setUploadStatus('1/2 단계: S3 Pre-signed URL 요청 중...');

        try {
            // UploadStartPayload 스키마에 맞춘 JSON 생성
            const payload = {
                upload_source: 'WEB_2D', // 필요에 따라 변경 (예: 'EXE_3D')
                original_filename: file.name,
                file_type: file.type || 'application/octet-stream',
                file_size_bytes: file.size,
                // non_member_identifier: 'optional-id' // 비회원 식별자가 있다면 추가
            };

            const response = await fetch(BACKEND_UPLOAD_API, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload),
                credentials: 'include', // 쿠키 기반 인증을 사용한다면 포함
            });

            if (!response.ok) {
                if (response.status === 401) {
                    alert('인증 세션이 만료되었습니다. 다시 로그인해주세요.');
                    router.push('/login');
                    return;
                }
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail?.message || `백엔드 오류: ${response.status}`);
            }

            const data = await response.json();
            // 백엔드에서 presigned_url과 s3_key를 반환하도록 되어 있으므로 그에 맞춰 사용
            const presignedUrl = data.presigned_url || data.url;
            const s3Key = data.s3_key || data.s3Key || data.s3_key;

            setUploadStatus('2/2 단계: S3에 파일 직접 업로드 중...');

            const s3UploadResponse = await fetch(presignedUrl, {
                method: 'PUT',
                headers: {
                    'Content-Type': file.type,
                },
                body: file,
            });

            if (s3UploadResponse.ok) {
                setUploadStatus(`🎉 업로드 성공! S3 키: ${s3Key}`);
            } else {
                throw new Error(`S3 업로드 실패: ${s3UploadResponse.statusText}`);
            }

        } catch (error) {
            console.error('업로드 실패 상세:', error);
            setUploadStatus(`❌ 업로드 실패: ${error instanceof Error ? error.message : String(error)}`);
        }
    };
    
    // ----------------------------------------------------
    // 3. UI 렌더링
    // ----------------------------------------------------

    return (
        <div style={{ padding: '50px', maxWidth: '600px', margin: 'auto', border: '1px solid #ddd', borderRadius: '8px' }}>
            <h1>🎥 비디오 업로드 (S3 Pre-signed URL)</h1>
            <p>현재 백엔드 URL: <code style={{ color: '#007bff' }}>{BACKEND_UPLOAD_API}</code></p>

            <div style={{ margin: '20px 0' }}>
                <input 
                    type="file" 
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept="*" 
                    style={{ display: 'block', marginBottom: '10px' }}
                />
            </div>

            <button
                onClick={handleUpload}
                disabled={!file || uploadStatus.includes('단계:')}
                style={{
                    padding: '10px 20px',
                    fontSize: '16px',
                    backgroundColor: (file ? '#007bff' : '#cccccc'),
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: (file ? 'pointer' : 'not-allowed')
                }}
            >
                {file ? `${file.name} 업로드 시작` : '파일을 선택하세요'}
            </button>

            <pre style={{ marginTop: '20px', padding: '10px', backgroundColor: '#f4f4f4', whiteSpace: 'pre-wrap' }}>
                {uploadStatus}
            </pre>
            
            <hr style={{ margin: '30px 0' }} />
            
            <button 
                onClick={() => router.push('/login')} 
                style={{ border: 'none', background: 'transparent', color: '#6c757d', cursor: 'pointer' }}
            >
                로그아웃 (토큰 세션 제거 필요)
            </button>
        </div>
    );
}