# backend-api/db_client.py

# MariaDB 드라이버 사용해 db 연결, 업로드 레코드 삽입

import os
import mysql.connector # 예시: MariaDB/MySQL 커넥터 사용 가정
from typing import Optional
from uuid import uuid4

# .env에서 DB 정보 로드
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_TABLE_NAME = os.getenv("DB_TABLE_NAME")

class DBClient:
    def __init__(self):
        try:
            self.conn = mysql.connector.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
        except Exception as e:
            print(f"DB 연결 실패: {e}")
            raise

    def insert_upload_intent(
        self, 
        job_id: str,    # job id를 외부에서 받아옴, websocket.py에서
        user_id: Optional[str],
        non_member_identifier: Optional[str], 
        upload_source: str,                   
        s3_key: str, 
        filename: str, 
        filetype: str, 
        file_size_bytes: int,     
        s3_result_path: Optional[str] = None    # 초기 result는 당연히 None
    ) -> str:
        """업로드 의도 레코드를 DB에 삽입하고, 생성된 UUID를 반환합니다."""
        
        upload_id = job_id  # 외부에서 받아온 job_id 사용

        sql = f"""
            INSERT INTO {DB_TABLE_NAME} (
                id, user_id, non_member_identifier, upload_source, 
                s3_key, original_filename, file_type, file_size_bytes, 
                processing_status, s3_result_path
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING', %s
            )
        """
        params = (
            upload_id, # job_id 사용
            user_id, 
            non_member_identifier, 
            upload_source,
            s3_key, 
            filename, 
            filetype, 
            file_size_bytes,
            s3_result_path
        )

        with self.conn.cursor() as cursor:
            cursor.execute(sql, params)
        self.conn.commit()
        return upload_id