**Autores: David García, Jaime Lozano y Marco Memba**

# Descripción del problema y datos

Necesitamos hacer un modelo que nos prediga si una reserva va a ser cancelada o no. Para ello analizaremos un dataset ubicado en la carpeta data/raw con las siguientes variables:

| Nombre Variable                  | Descripción                                              |
| -------------------------------- | -------------------------------------------------------- |
| `hotel`                          | Tipo de hotel: City Hotel o Resort Hotel                 |
| `is_canceled`                    | Variable objetivo: 1 si fue cancelado, 0 si no           |
| `lead_time`                      | Días entre la reserva y la fecha de llegada              |
| `arrival_date_year`              | Año de llegada                                           |
| `arrival_date_month`             | Mes de llegada                                           |
| `arrival_date_week_number`       | Número de la semana del año                              |
| `arrival_date_day_of_month`      | Día del mes de llegada                                   |
| `stays_in_weekend_nights`        | Noches de fin de semana reservadas                       |
| `stays_in_week_nights`           | Noches entre semana reservadas                           |
| `adults`                         | Número de adultos                                        |
| `children`                       | Número de niños                                          |
| `babies`                         | Número de bebés                                          |
| `meal`                           | Tipo de comida reservada                                 |
| `country`                        | País de origen del cliente                               |
| `market_segment`                 | Canal de marketing (online, offline, grupos...)          |
| `distribution_channel`           | Canal de distribución (directo, TA/TO...)                |
| `is_repeated_guest`              | 1 si el cliente ha estado anteriormente                  |
| `previous_cancellations`         | Nº de cancelaciones anteriores                           |
| `previous_bookings_not_canceled` | Nº de reservas previas no canceladas                     |
| `reserved_room_type`             | Tipo de habitación reservada                             |
| `assigned_room_type`             | Tipo de habitación asignada                              |
| `booking_changes`                | Nº de cambios en la reserva                              |
| `deposit_type`                   | Tipo de depósito: No Deposit, Refundable, etc.           |
| `agent`                          | ID del agente (puede ser nulo)                           |
| `company`                        | ID de la empresa (puede ser nulo)                        |
| `days_in_waiting_list`           | Días en lista de espera                                  |
| `customer_type`                  | Tipo de cliente: Transient, Group, etc.                  |
| `adr`                            | Average Daily Rate (precio promedio por noche)           |
| `required_car_parking_spaces`    | Plazas de parking solicitadas                            |
| `total_of_special_requests`      | Nº de peticiones especiales                              |
| `reservation_status`             | Estado final de la reserva: Check-Out, Canceled, No-Show |
| `reservation_status_date`        | Fecha en que se actualizó el estado                      |

# Instrucciones para ejecutar el proyecto

## 1. Requisitos previos

Es necesario tener ``Python`` instalado y actualizado. También será necesario tener la librería ``pip`` y ``uv``.

## 2. Crear y activar entorno virtual

Ejecutar en un terminal:

```bash
# Mac/Linux
uv venv --python 3.12
source .venv/bin/activate

# Windows
uv venv --python 3.12
.\.venv\Scripts\activate
```

## 3. Instalar dependencias

Ejecutar en un terminal:

```bash
uv pip install -r requirements.txt
```

## 4. Ejecutar la aplicación

Ejecutar en un terminal situado en la raíz del proyecto:

```bash
python main.py
```

## 5. Desactivar entorno virtual

Ejecutar en un terminal situado en la raíz del proyecto o en alguno de sus subdirectorios:

```bash
deactivate
```

# Resultados y conclusiones

| Modelo | Accuracy | Precisión | Recall | F1-score | AUC |
|---|---|---|---|---|---|
| **LightGBM** | 0.892663 | 0.869529 | 0.835613 | 0.852234 | 0.959873 |
| **Random Forest** | 0.865692 | 0.905728 | 0.711475 | 0.796935 | 0.947074 |
| **Decision Tree** | 0.864813 | 0.825623 | 0.805088 | 0.815226 | 0.891259 |
| **Red neuronal Keras** | 0.863473 | 0.846163 | 0.771735 | 0.807237 | 0.941123 |
| **Logistic Regression** | 0.823185 | 0.801487 | 0.694743 | 0.744307 | 0.905255 |

**El algoritmo ganador es LightGBM**. Es superior a todos los demás en 4 de las 5 métricas: Accuracy, Recall, F1-score y AUC LightGBM consigue 87% de predicciones de cancelación correctas (Precisión) y detecta el 84% de todas las cancelaciones reales (Recall).

La comparación entre los modelos que usan árboles (Decision Tree y su evolución Random Forest) arrojan los resultados esperados: Random Forest es superior en Accuracy, Precisión y AUC.

La red neuronal multicapa usando Keras se queda por debajo de LightGBM y de Random Forest, demostrando que el boosting es superior a las redes neuronales en este dataset con datos tabulares.

También se observa la coherencia de resultados en Logistic Regression, ocupando el último lugar de la tabla en 4 de 5 métricas, únicamente siendo superior a Decision Tree en AUC. Aún así, es muy superior a un clasificador que siempre prediga is_canceled = 0 el cual obtendría 63% Accuracy y 0,5 AUC.
