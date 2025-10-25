app/ 폴더 안에 웹 페이지 파일(tsx)존재

login/ 폴더는 OAuth 창 리다이렉트 하는 페이지

callback/ 은 OAuth 성공후 받은 code를 /token api로 보내 JWT token으로 교환함
/token api 백엔드는 token을 http only cookie에 저장