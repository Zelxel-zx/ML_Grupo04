import os

import requests


class SupabaseClient:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.key = os.getenv("SUPABASE_KEY", "")

        if not self.url or not self.key:
            raise RuntimeError(
                "Faltan variables de entorno SUPABASE_URL y SUPABASE_KEY."
            )

        self.base_url = f"{self.url}/rest/v1"
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Accept": "application/json",
        }

    def select(self, table, params=None):
        query_params = {"select": "*"}
        if params:
            query_params.update(params)

        response = requests.get(
            f"{self.base_url}/{table}",
            headers=self.headers,
            params=query_params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
