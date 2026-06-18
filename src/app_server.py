from __future__ import annotations

import json
import pickle
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
HOST = "127.0.0.1"
PORT = 8000


RECURSOS_POR_PRIORIDAD = {
    "Alta": [
        "Metotrexato 500mg",
        "Suero Fisiologico 500ml",
        "Dexametasona 4mg",
        "Pediasure Plus 200ml",
        "Huevo de gallina",
        "Leche evaporada",
        "Pechuga de pollo",
    ],
    "Media": [
        "Metotrexato 500mg",
        "Suero Fisiologico 500ml",
        "Pediasure Plus 200ml",
        "Leche evaporada",
        "Huevo de gallina",
    ],
    "Baja": [
        "Suero Fisiologico 500ml",
        "Pediasure Plus 200ml",
        "Leche evaporada",
        "Huevo de gallina",
    ],
}


def load_pickle(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def sanitize_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


class PredictionService:
    def __init__(self) -> None:
        self.clin_payload = load_pickle(DATA_DIR / "modelo_clinico.pkl")
        self.log_payload = load_pickle(DATA_DIR / "modelo_logistico.pkl")
        self.clin_metrics = json.loads((DATA_DIR / "resultados_modelo_clinico.json").read_text(encoding="utf-8"))
        self.log_metrics = json.loads((DATA_DIR / "resultados_modelo_logistico.json").read_text(encoding="utf-8"))
        self.patients_df = pd.read_json(DATA_DIR / "dataset_clinico_app.json")
        self.patients_df = self.patients_df[self.patients_df["estado_vital"] != "Fallecido"].copy()
        self.log_base_df = pd.read_csv(DATA_DIR / "dataset_logistico_1.csv")
        self._prepare_logistic_base()

    def _prepare_logistic_base(self) -> None:
        df = self.log_base_df.copy()
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

        for base in [
            "product_id",
            "store_id",
            "first_category_id",
            "second_category_id",
            "third_category_id",
            "holiday_flag",
            "activity_flag",
        ]:
            code_col = f"{base}_cod"
            df[base] = df[base].astype("category")
            df[code_col] = df[base].cat.codes

        numeric_fill = {}
        for feat in self.log_payload["features"]:
            if feat in df.columns:
                numeric_fill[feat] = float(df[feat].median()) if pd.api.types.is_numeric_dtype(df[feat]) else 0
                df[feat] = df[feat].fillna(numeric_fill[feat])
            else:
                numeric_fill[feat] = 0
                df[feat] = 0

        self.log_fill_values = numeric_fill
        self.log_base_df = df

    def get_patients(self):
        columns = [
            "paciente_id",
            "grupo_edad",
            "sexo",
            "raza",
            "estadio",
            "estado_vital",
            "supervivencia_meses",
            "prioridad_real",
            "prioridad_predicha",
            "confianza_prediccion",
            "foco_clinico",
        ]
        return self.patients_df[columns].to_dict(orient="records")

    def _get_patient_row(self, patient_id: str) -> pd.Series:
        match = self.patients_df[self.patients_df["paciente_id"] == patient_id]
        if match.empty:
            raise KeyError(f"Paciente no encontrado: {patient_id}")
        return match.iloc[0]

    def predict_clinical(self, patient_id: str):
        patient = self._get_patient_row(patient_id)
        feature_values = {feat: patient[f"model__{feat}"] for feat in self.clin_payload["features"]}
        X = pd.DataFrame([feature_values], columns=self.clin_payload["features"])
        pred = self.clin_payload["modelo"].predict(X)[0]
        confidence = None
        if hasattr(self.clin_payload["modelo"], "predict_proba"):
            proba = self.clin_payload["modelo"].predict_proba(X)[0]
            confidence = float(proba.max())

        return {
            "paciente_id": patient_id,
            "prioridad_predicha": pred,
            "confianza_prediccion": confidence,
            "foco_clinico": "Cancer de medula osea",
        }

    def _predict_logistic_for_resources(self, resource_names: list[str]):
        subset = (
            self.log_base_df[self.log_base_df["name_products"].isin(resource_names)]
            .sort_values(["name_products", "dt"])
            .groupby("name_products", as_index=False)
            .tail(1)
            .copy()
        )
        if subset.empty:
            return []

        X = subset[self.log_payload["features"]].copy()
        for feat, fill_value in self.log_fill_values.items():
            X[feat] = X[feat].fillna(fill_value)
        subset["prediccion_demanda"] = self.log_payload["modelo"].predict(X)
        subset["stock_actual_referencia"] = subset["stock_lag_1"].fillna(subset["horas_con_stock"])
        subset["cobertura_estimada"] = subset["stock_actual_referencia"] / subset["prediccion_demanda"].clip(lower=1)
        subset["riesgo_quiebre"] = subset.apply(self._risk_level, axis=1)
        subset["dt"] = subset["dt"].dt.strftime("%Y-%m-%d")

        cols = [
            "dt",
            "name_products",
            "tipo_producto_app",
            "prediccion_demanda",
            "stock_actual_referencia",
            "cobertura_estimada",
            "horas_con_stock",
            "riesgo_quiebre",
        ]
        records = []
        for record in subset[cols].to_dict(orient="records"):
            records.append({k: sanitize_value(v) for k, v in record.items()})
        return records

    @staticmethod
    def _risk_level(row) -> str:
        cobertura = float(row["cobertura_estimada"] or 0)
        if cobertura < 0.08:
            return "Alto"
        if cobertura < 0.14:
            return "Medio"
        return "Bajo"

    @staticmethod
    def _highest_risk(resources: list[dict]) -> str:
        levels = [item["riesgo_quiebre"] for item in resources]
        if "Alto" in levels:
            return "Alto"
        if "Medio" in levels:
            return "Medio"
        return "Bajo"

    def integrated_prediction(self, patient_id: str):
        patient = self._get_patient_row(patient_id)
        clinical = self.predict_clinical(patient_id)
        prioridad = clinical["prioridad_predicha"]
        resources = self._predict_logistic_for_resources(RECURSOS_POR_PRIORIDAD.get(prioridad, RECURSOS_POR_PRIORIDAD["Baja"]))
        demanda_total = round(sum(float(item["prediccion_demanda"]) for item in resources), 2)
        riesgo = self._highest_risk(resources) if resources else "Bajo"

        return {
            "paciente": {
                "paciente_id": patient["paciente_id"],
                "grupo_edad": patient["grupo_edad"],
                "sexo": patient["sexo"],
                "raza": patient["raza"],
                "estadio": patient["estadio"],
                "estado_vital": patient["estado_vital"],
                "supervivencia_meses": sanitize_value(patient["supervivencia_meses"]),
                "prioridad_real": patient["prioridad_real"],
            },
            "clinico": {
                **clinical,
                "metricas_modelo": self.clin_metrics["metricas"],
            },
            "logistico": {
                "demanda_total_proyectada": demanda_total,
                "riesgo_operativo": riesgo,
                "metricas_modelo": self.log_metrics["metricas"],
                "recursos": resources,
            },
        }


SERVICE = PredictionService()


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def _send_json(self, payload: dict | list, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return self._send_json({"status": "ok"})
        if parsed.path == "/api/patients":
            return self._send_json(SERVICE.get_patients())
        if parsed.path == "/api/integrated":
            patient_id = parse_qs(parsed.query).get("patient_id", [""])[0]
            if not patient_id:
                return self._send_json({"error": "patient_id es requerido"}, status=400)
            try:
                return self._send_json(SERVICE.integrated_prediction(patient_id))
            except KeyError as exc:
                return self._send_json({"error": str(exc)}, status=404)
        return super().do_GET()


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}/src/dashborad.html")


def run():
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Servidor listo en http://{HOST}:{PORT}/src/dashborad.html")
    threading.Timer(1.0, open_browser).start()
    server.serve_forever()


if __name__ == "__main__":
    run()
