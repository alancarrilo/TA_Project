# funciones.py

import itertools
import optuna
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, CCIIndicator
from ta.volatility import BollingerBands
import numpy as np


def cargar_datos(ruta_entrenamiento, ruta_prueba):
    entrenamiento = pd.read_csv(ruta_entrenamiento).dropna()
    prueba = pd.read_csv(ruta_prueba).dropna()
    return entrenamiento, prueba


def generar_combinaciones(indicadores):
    todas_combinaciones = []
    for r in range(1, len(indicadores) + 1):
        combinaciones = itertools.combinations(indicadores, r)
        todas_combinaciones.extend(combinaciones)
    return todas_combinaciones


def backtest(datos, estrategia, stop_perdida, tomar_ganancia, n_acciones, capital_inicial, comision,
             rsi_window=None, rsi_lower=None, rsi_upper=None,
             bollinger_window=None, bollinger_stddev=None,
             macd_fast=None, macd_slow=None, macd_signal=None,
             cci_window=None, stochastic_k=None, stochastic_d=None,
             tasa_libre_riesgo=0.01):
    # Calcular los indicadores técnicos
    if "RSI" in estrategia and rsi_window is not None:
        rsi = RSIIndicator(close=datos['Close'], window=rsi_window).rsi()
    else:
        rsi = pd.Series(index=datos.index)

    if "Bollinger" in estrategia and bollinger_window is not None and bollinger_stddev is not None:
        bollinger = BollingerBands(close=datos['Close'], window=bollinger_window, window_dev=bollinger_stddev)
    else:
        bollinger = None

    if "MACD" in estrategia and macd_fast is not None and macd_slow is not None and macd_signal is not None:
        macd = MACD(close=datos['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
    else:
        macd = None

    if "CCI" in estrategia and cci_window is not None:
        cci = CCIIndicator(high=datos['High'], low=datos['Low'], close=datos['Close'], window=cci_window).cci()
    else:
        cci = pd.Series(index=datos.index)

    if "Stochastic" in estrategia and stochastic_k is not None and stochastic_d is not None:
        stochastic = StochasticOscillator(high=datos['High'], low=datos['Low'], close=datos['Close'],
                                          window=stochastic_k, smooth_window=stochastic_d)
    else:
        stochastic = None

    # Variables del capital y posición inicial
    capital = capital_inicial
    posicion = 0
    precio_compra = 0

    # Para el cálculo de métricas adicionales
    valores_portafolio = [capital_inicial]
    rendimientos = []
    max_valor = capital_inicial
    max_drawdown = 0
    operaciones_ganadoras = 0
    operaciones_perdedoras = 0
    n_operaciones = 0

    # Iterar sobre los datos usando índices posicionales
    for i in range(len(datos)):
        fila = datos.iloc[i]

        # Señales de indicadores
        rsi_senal = bollinger_senal = macd_senal = cci_senal = stochastic_senal = 0

        if "RSI" in estrategia and not np.isnan(rsi.iloc[i]):
            if rsi.iloc[i] < rsi_lower:
                rsi_senal = 1
            elif rsi.iloc[i] > rsi_upper:
                rsi_senal = -1

        if "Bollinger" in estrategia and bollinger is not None:
            lband = bollinger.bollinger_lband().iloc[i]
            hband = bollinger.bollinger_hband().iloc[i]
            if not np.isnan(lband) and not np.isnan(hband):
                if fila['Close'] < lband:
                    bollinger_senal = 1
                elif fila['Close'] > hband:
                    bollinger_senal = -1

        if "MACD" in estrategia and macd is not None:
            macd_diff = macd.macd_diff().iloc[i]
            if not np.isnan(macd_diff):
                if macd_diff > 0:
                    macd_senal = 1
                elif macd_diff < 0:
                    macd_senal = -1

        if "CCI" in estrategia and not np.isnan(cci.iloc[i]):
            if cci.iloc[i] < -100:
                cci_senal = 1
            elif cci.iloc[i] > 100:
                cci_senal = -1

        if "Stochastic" in estrategia and stochastic is not None:
            stoch_value = stochastic.stoch().iloc[i]
            if not np.isnan(stoch_value):
                if stoch_value < 20:
                    stochastic_senal = 1
                elif stoch_value > 80:
                    stochastic_senal = -1

        # Estrategia de compra/venta simple
        senal_total = rsi_senal + bollinger_senal + macd_senal + cci_senal + stochastic_senal
        if senal_total > 0 and capital >= fila['Close'] * n_acciones:
            posicion += n_acciones
            precio_compra = fila['Close']
            capital -= fila['Close'] * n_acciones * (1 + comision)
            n_operaciones += 1
        elif senal_total < 0 and posicion > 0:
            ganancia_operacion = (fila['Close'] - precio_compra) * n_acciones
            if ganancia_operacion > 0:
                operaciones_ganadoras += 1
            else:
                operaciones_perdedoras += 1
            capital += fila['Close'] * n_acciones * (1 - comision)
            posicion -= n_acciones
            n_operaciones += 1

        # Stop loss / Take profit
        if posicion > 0:
            if fila['Close'] <= precio_compra * (1 - stop_perdida) or fila['Close'] >= precio_compra * (
                    1 + tomar_ganancia):
                ganancia_operacion = (fila['Close'] - precio_compra) * posicion
                if ganancia_operacion > 0:
                    operaciones_ganadoras += 1
                else:
                    operaciones_perdedoras += 1
                capital += fila['Close'] * posicion * (1 - comision)
                posicion = 0
                n_operaciones += 1

        # Actualizar valor del portafolio y métricas
        valor_portafolio = capital + posicion * fila['Close']
        if not np.isnan(valor_portafolio):
            valores_portafolio.append(valor_portafolio)
            if len(valores_portafolio) > 1:
                rendimiento = (valor_portafolio - valores_portafolio[-2]) / valores_portafolio[-2]
                rendimientos.append(rendimiento)

            # Máximo Drawdown
            if valor_portafolio > max_valor:
                max_valor = valor_portafolio
            drawdown = (max_valor - valor_portafolio) / max_valor
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    # Valor final del portafolio
    valor_final = capital + posicion * datos['Close'].iloc[-1]

    # Cálculo del Sharpe Ratio
    rendimientos = np.array(rendimientos)
    rendimientos = rendimientos[~np.isnan(rendimientos)]  # Eliminar posibles NaN
    if len(rendimientos) > 1 and np.std(rendimientos) != 0:
        exceso_retornos = rendimientos - tasa_libre_riesgo / 252  # Asumiendo 252 días de trading
        sharpe_ratio = np.mean(exceso_retornos) / np.std(exceso_retornos) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Win-Loss Ratio
    if operaciones_perdedoras > 0:
        win_loss_ratio = operaciones_ganadoras / operaciones_perdedoras
    elif operaciones_ganadoras > 0:
        win_loss_ratio = np.inf  # Si no hay operaciones perdedoras
    else:
        win_loss_ratio = 0  # Si no hay operaciones ganadoras ni perdedoras

    resultados = {
        'valor_final': valor_final,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_loss_ratio': win_loss_ratio,
        'n_operaciones': n_operaciones,
        'valores_portafolio': valores_portafolio  # Agregar el historial del portafolio
    }

    return resultados


def opt(trial, datos, estrategia, capital_inicial, comision, tasa_libre_riesgo=0.01):
    # Sugerencias de hiperparámetros mediante Optuna
    stop_perdida = trial.suggest_float("stop_perdida", 0.01, 0.1)
    tomar_ganancia = trial.suggest_float("tomar_ganancia", 0.01, 0.2)
    n_acciones = trial.suggest_int("n_acciones", 1, 100)

    # Parámetros para los indicadores técnicos
    rsi_window = rsi_lower = rsi_upper = None
    bollinger_window = bollinger_stddev = None
    macd_fast = macd_slow = macd_signal = None
    cci_window = None
    stochastic_k = stochastic_d = None

    if "RSI" in estrategia:
        rsi_window = trial.suggest_int("rsi_window", 5, 30)
        rsi_lower = trial.suggest_float("rsi_lower", 20, 50)
        rsi_upper = trial.suggest_float("rsi_upper", 50, 80)

    if "Bollinger" in estrategia:
        bollinger_window = trial.suggest_int("bollinger_window", 10, 50)
        bollinger_stddev = trial.suggest_float("bollinger_stddev", 1.5, 3.5)

    if "MACD" in estrategia:
        macd_fast = trial.suggest_int("macd_fast", 12, 26)
        macd_slow = trial.suggest_int("macd_slow", 26, 50)
        macd_signal = trial.suggest_int("macd_signal", 9, 18)

    if "CCI" in estrategia:
        cci_window = trial.suggest_int("cci_window", 10, 40)

    if "Stochastic" in estrategia:
        stochastic_k = trial.suggest_int("stochastic_k", 5, 20)
        stochastic_d = trial.suggest_int("stochastic_d", 3, 14)

    # Ejecutar backtest con los parámetros optimizados
    resultado = backtest(datos, estrategia, stop_perdida, tomar_ganancia, n_acciones, capital_inicial, comision,
                         rsi_window, rsi_lower, rsi_upper,
                         bollinger_window, bollinger_stddev,
                         macd_fast, macd_slow, macd_signal,
                         cci_window, stochastic_k, stochastic_d,
                         tasa_libre_riesgo)

    # Imprimir las métricas después de cada trial
    print(f"Trial {trial.number}, Estrategia: {estrategia}")
    print(f"Parámetros: {trial.params}")
    print(f"Valor final: {resultado['valor_final']}")
    print(f"Sharpe Ratio: {resultado['sharpe_ratio']}")
    print(f"Máximo Drawdown: {resultado['max_drawdown']}")
    print(f"Win-Loss Ratio: {resultado['win_loss_ratio']}")
    print(f"Número de operaciones: {resultado['n_operaciones']}\n")

    # Optuna intentará maximizar el valor final del portafolio
    return resultado['valor_final']


def ejecutar_optimizacion(datos_entrenamiento, datos_prueba, combinaciones, capital_inicial, comision, n_trials=100,
                          tasa_libre_riesgo=0.01):
    mejor_combinacion = None
    mejor_valor = -float('inf')
    resultados = []

    # Para cada combinación de indicadores técnicos
    for estrategia in combinaciones:
        print(f"\nOptimizando para la combinación: {estrategia}")

        # Crear un estudio de Optuna para maximizar el valor del portafolio
        estudio = optuna.create_study(direction="maximize")

        # Ejecutar la optimización con Optuna en los datos de entrenamiento
        estudio.optimize(
            lambda trial: opt(trial, datos_entrenamiento, estrategia, capital_inicial, comision, tasa_libre_riesgo),
            n_trials=n_trials)

        # Guardar los resultados de cada combinación
        mejores_parametros = estudio.best_params

        # Ejecutar backtest con los mejores parámetros para obtener todas las métricas
        resultado_entrenamiento = backtest(datos_entrenamiento, estrategia,
                                           mejores_parametros['stop_perdida'], mejores_parametros['tomar_ganancia'],
                                           mejores_parametros['n_acciones'], capital_inicial, comision,
                                           rsi_window=mejores_parametros.get('rsi_window'),
                                           rsi_lower=mejores_parametros.get('rsi_lower'),
                                           rsi_upper=mejores_parametros.get('rsi_upper'),
                                           bollinger_window=mejores_parametros.get('bollinger_window'),
                                           bollinger_stddev=mejores_parametros.get('bollinger_stddev'),
                                           macd_fast=mejores_parametros.get('macd_fast'),
                                           macd_slow=mejores_parametros.get('macd_slow'),
                                           macd_signal=mejores_parametros.get('macd_signal'),
                                           cci_window=mejores_parametros.get('cci_window'),
                                           stochastic_k=mejores_parametros.get('stochastic_k'),
                                           stochastic_d=mejores_parametros.get('stochastic_d'),
                                           tasa_libre_riesgo=tasa_libre_riesgo)

        resultados.append({
            "estrategia": estrategia,
            "mejores_parametros": mejores_parametros,
            "resultado_entrenamiento": resultado_entrenamiento
        })

        # Si esta combinación es la mejor hasta ahora, la guardamos
        if estudio.best_value > mejor_valor:
            mejor_valor = estudio.best_value
            mejor_combinacion = estrategia

    if mejor_combinacion is None:
        print("No se encontró ninguna estrategia viable.")
        return None, resultados, None

    # Validar la mejor combinación en los datos de prueba
    print("\nValidando la mejor combinación en los datos de prueba...")
    mejores_parametros = next(resultado for resultado in resultados if resultado['estrategia'] == mejor_combinacion)[
        'mejores_parametros']

    # Ejecutar backtest con los datos de prueba usando los mejores parámetros
    resultado_prueba = backtest(datos_prueba, mejor_combinacion,
                                mejores_parametros['stop_perdida'], mejores_parametros['tomar_ganancia'],
                                mejores_parametros['n_acciones'], capital_inicial, comision,
                                rsi_window=mejores_parametros.get('rsi_window'),
                                rsi_lower=mejores_parametros.get('rsi_lower'),
                                rsi_upper=mejores_parametros.get('rsi_upper'),
                                bollinger_window=mejores_parametros.get('bollinger_window'),
                                bollinger_stddev=mejores_parametros.get('bollinger_stddev'),
                                macd_fast=mejores_parametros.get('macd_fast'),
                                macd_slow=mejores_parametros.get('macd_slow'),
                                macd_signal=mejores_parametros.get('macd_signal'),
                                cci_window=mejores_parametros.get('cci_window'),
                                stochastic_k=mejores_parametros.get('stochastic_k'),
                                stochastic_d=mejores_parametros.get('stochastic_d'),
                                tasa_libre_riesgo=tasa_libre_riesgo)

    print(f"\nResultados del conjunto de prueba para la mejor combinación ({mejor_combinacion}):")
    print(f"Valor final del portafolio: {resultado_prueba['valor_final']}")
    print(f"Sharpe Ratio: {resultado_prueba['sharpe_ratio']}")
    print(f"Máximo Drawdown: {resultado_prueba['max_drawdown']}")
    print(f"Win-Loss Ratio: {resultado_prueba['win_loss_ratio']}")
    print(f"Número de operaciones: {resultado_prueba['n_operaciones']}")

    return mejor_combinacion, resultados, resultado_prueba
