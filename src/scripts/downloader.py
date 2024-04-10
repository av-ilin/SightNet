from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload

import os

class GoogleDriveDownloader:
    def __init__(self):
        self.service = self._authenticate()

    def _authenticate(self):
        SCOPES = ['https://www.googleapis.com/auth/drive']
        SERVICE_ACCOUNT_FILE = 'cv-sine-4370c3fb1e35.json'  # Путь к вашему файлу учетных данных
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return build('drive', 'v3', credentials=credentials)

    def download_folder_contents(self, folder_id, destination_path):
        results = self.service.files().list(q=f"'{folder_id}' in parents", fields="files(id, name, mimeType)").execute()
        items = results.get('files', [])

        for item in items:
            file_id = item['id']
            file_name = item['name']
            file_type = item['mimeType']

            if file_type == 'application/vnd.google-apps.folder':
                subfolder_path = os.path.join(destination_path, file_name)
                os.makedirs(subfolder_path, exist_ok=True)
                self.download_folder_contents(file_id, subfolder_path)
            else:
                request = self.service.files().get_media(fileId=file_id)
                file_path = os.path.join(destination_path, file_name)
                with open(file_path, 'wb') as file:
                    downloader = MediaIoBaseDownload(file, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        print(f"Downloaded {int(status.progress() * 100)}% of {file_name}")

if __name__ == "__main__":
    downloader = GoogleDriveDownloader()
    folder_id = '1wDkSNTEAZf0DJwRaXlQAmMIj4JIEmNYS' 
    destination_path = 'downloaded_files' 
    downloader.download_folder_contents(folder_id, destination_path)