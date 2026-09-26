Autores: David García, Jaime Lozano y Marco Memba

Descripción del problema y datos

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

Instrucciones para ejecutar el proyecto

MacOS

Se tendrá que ejecutar uv sync en la terminal de comandos. De esta forma, se descargarán todas las librerías que necesitaremos para poder ejecutar el código.
Una vez instaladas las librerías tenemos que arrancar el entorno virtual. Para ello ejecutaremos source .venv/bin/active también en la terminal de comandos.
Y para ejecutar la app desde la línea de comandos situandonos en la raíz del proyecto escribimos uv run main.py.

Windows

Se tendrá que ejecutar uv sync en la terminal de comandos. De esta forma, se descargarán todas las librerías que necesitaremos para poder ejecutar el código.
Una vez instaladas las librerías tenemos que arrancar el entorno virtual. Para ello ejecutaremos .\env-pontia-ml\Scripts\activate también en la terminal de comandos.
Y para ejecutar la app desde la línea de comandos situandonos en la raíz del proyecto escribimos uv run main.py.

Resultados y conclusiones

Para poder escoger un modelo nos hemos basado en las métrica Accurancy y nos salio con mejor métrica el modelo LightBGM. En otras pruebas nos salio mejor modelo el Random Forest mirando meétricas como el Accuracy o el AUC. 
Nos hemos basado en estas métricas porque era un problema de clasificación binaria y las métricas que nos dan valor son el Accuracy, el F1-Score y el AUC. Tanto el Accuracy como el AUC nos han parecido métricas más explicables porque en el AUC podíamos apoyarnos en gráficos. También hemos pensamos que los resultados no eran tan sensibles a posibles errores como podría ser un diagnostico clínico.


| Modelo | Accuracy | Precisión | Recall | F1 | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 0.890108 | 0.864782 | 0.833691 | 0.848952 | 0.959048 |
| **Decision Tree** | 0.864687 | 0.830702 | 0.797174 | 0.813593 | 0.922536 |
| **Keras Neural Net** | 0.864687 | 0.844586 | 0.777841 | 0.809841 | 0.940740 |
| **Gradient Boosting** | 0.852835 | 0.861032 | 0.718711 | 0.783461 | 0.929908 |
| **Logistic Regression** | 0.817363 | 0.810612 | 0.661504 | 0.728507 | 0.900985 |
| **Random Forest** | 0.781933 | 0.969056 | 0.424873 | 0.590741 | 0.915823 |
