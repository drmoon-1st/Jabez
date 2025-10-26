## 페이지

app/ 폴더 안에 웹 페이지 파일(tsx)존재

login/ 폴더는 OAuth 창 리다이렉트 하는 페이지

callback/ 은 OAuth 성공후 받은 code를 /token api로 보내 JWT token으로 교환함  
/token api 백엔드는 token을 http only cookie에 저장

uploader/ 는 업로드 파일 선택후 업로드 프로세스 시작  
업로드 프로세스는 /api/upload 에서 요구하는 정보들을 json으로 보냄

## 업로드 json api 형식  
{  
upload_source: WEB_2D (웹에서 보내는 2d 영상)  
original_filename: 원래 파일 이름  
file_type: image/jpeg, video/mp4 등, 현재는 테스트 위해 파일 종류 제한 막지 않음, 실 서비스 시에는 오직 비디오로만 제한  
file_size_bytes: 파일 사이즈, 자동으로 계산됨  
non_member_identifier: 비회원 식별자, 현재는 비회원 기능은 없음  
}  

/api/upload에서는 json 받고 s3 전송용 presinged url 리턴
이를 사용해 s3로 직접 업로드
