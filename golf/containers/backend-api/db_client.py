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
        user_id: Optional[str],
        non_member_identifier: Optional[str], 
        upload_source: str,                   
        s3_key: str, 
        filename: str, 
        filetype: str, 
        file_size_bytes: int,                 
    ) -> str:
        """업로드 의도 레코드를 DB에 삽입하고, 생성된 UUID를 반환합니다."""
        
        # FastAPI에서 생성된 UUID를 사용하거나, DB에서 생성하도록 설정할 수 있습니다.
        upload_id = str(uuid4()) 
        
        sql = """
            INSERT INTO uploads (
                id, user_id, non_member_identifier, upload_source, 
                s3_key, original_filename, file_type, file_size_bytes, 
                processing_status
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING'
            )
        """
        params = (
            upload_id, 
            user_id, 
            non_member_identifier, 
            upload_source,
            s3_key, 
            filename, 
            filetype, 
            file_size_bytes
        )

        with self.conn.cursor() as cursor:
            cursor.execute(sql, params)
        self.conn.commit()
        return upload_id