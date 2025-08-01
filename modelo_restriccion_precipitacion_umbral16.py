
import numpy as np
import pandas as pd

class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out):
        self.IW = IW
        self.bias_IW = bias_IW
        self.LW = LW
        self.bias_out = bias_out
        self.input_min = np.array([1, 6.4, -6.3, 0])
        self.input_max = np.array([301, 40.4, 23.7, 106])
        self.umbral_emeac = 16

    def tansig(self, x):
        return np.tanh(x)

    def normalize_input(self, X_real):
        return 2 * (X_real - self.input_min) / (self.input_max - self.input_min) - 1

    def desnormalize_output(self, y_norm, ymin=-1, ymax=1):
        return (y_norm - ymin) / (ymax - ymin)

    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2)

    def predict(self, X_real, fechas, prec):
        X_norm = self.normalize_input(X_real)
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm])
        emerrel_desnorm = self.desnormalize_output(emerrel_pred)

        # Filtro de precipitación acumulada en 8 días >= 5 mm
        prec_acum_8 = np.convolve(prec, np.ones(8, dtype=int), mode='full')[:len(prec)]
        filtro_prec = prec_acum_8 >= 5
        emerrel_filtrado = emerrel_desnorm.copy()
        emerrel_filtrado[~filtro_prec] = 0

        emeac = np.cumsum(emerrel_filtrado) / self.umbral_emeac
        emeac_pct = np.clip(emeac * 100, 0, 100)

        return pd.DataFrame({
            "Fecha": fechas,
            "EMERREL (0-1)": emerrel_filtrado,
            "EMEAC (%)": emeac_pct
        })
