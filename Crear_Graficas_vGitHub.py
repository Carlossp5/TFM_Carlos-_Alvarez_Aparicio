pip install matplotlib

# 0. Importar librerias

import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

# Cargo tambien el repositorio de Github donde subiré todo lo necesario para productivizar la app web
repo_path = "https://github.com/Carlossp5/TFM_Carlos-_Alvarez_Aparicio.git"

# 1. Lectura de ficheros
# Leo las tablas anteriormente guardadas
GRANDES_5_LIGAS_EUR_FBREF_201617_202223 = pd.read_csv("https://raw.githubusercontent.com/Carlossp5/TFM_Carlos-_Alvarez_Aparicio/main/Datasets/GRANDES_5_LIGAS_EUR_FBREF_201617_202223_std.csv", sep=',', index_col=False)
GRANDES_5_LIGAS_EUR_FBREF_202324_J28 = pd.read_csv("https://raw.githubusercontent.com/Carlossp5/TFM_Carlos-_Alvarez_Aparicio/main/Datasets/GRANDES_5_LIGAS_EUR_FBREF_202324_J28_std.csv", sep=',', index_col=False)
CHAMP_GRANDES_5_LIGAS_EUR_FBREF_201617_202223 = pd.read_csv("https://raw.githubusercontent.com/Carlossp5/TFM_Carlos-_Alvarez_Aparicio/main/Datasets/CHAMP_GRANDES_5_LIGAS_EUR_FBREF_201617_202223_std.csv", sep=',', index_col=False)

# 2. Definicion de los input de la app
# Defino los diferentes paquetes de variables. Mismos grupos que usa FBREF
Clasificacion = ['MP','W','D','L','GF','GA','GD','Pts','Pts/MP','xG','xGA','xGD','xGD/90']
Clasificacion_Home = ['Rk_Home','Partidos_Home','W_Home','D_Home','L_Home','GF_Home','GA_Home','GD_Home','Pts_Home','Pts_Partido_Home','xG_Home','xGA_Home','xGD_Home','xGD/90_Home']
Clasificacion_Away = ['Partidos_Away','W_Away','D_Away','L_Away','GF_Away','GA_Away','GD_Away','Pts_Away','Pts_Partido_Away','xG_Away',	'xGA_Away','xGD_Away','xGD/90_Away']
Standard_Stats_Own = ['Jugadores_Involucrados','Age_media_equipo_propio','Posesion_media_a_favor ','Ast_a_favor','G+A_a_favor','Goles_Sin_Penaltis_a_favor','PK','Penaltis_Intentados_a_favor',
                      'Tarjetas_Amarillas_propias','Tarjetas_Rojas_propias','npxG','xAst_a_favor','npxG+xAG_a_favor','Prog_con_Conduccion_a_favor','Prog_con_Pase_a_favor','Gls/90_a_favor',
                      'Ast/90_a_favor','G+A.1_a_favor','G-PK/90_a_favor','xG/90_a_favor','xAst/90_a_favor','xG+xAG/90_a_favor','npxG/90_a_favor','npxG+xAst/90_a_favor']
Standard_Stats_Against = ['Age_media_equipos_rivales','Posesion_media_en_contra','Ast_en_contra','G+A_en_contra','Goles_Sin_Penaltis_en_contra','PK_x','Penaltis_Intentados_en_contra',
                          'Tarjetas_Amarillas_en_contra','Tarjetas_Rojas_en_contra','npxG_x','xAst_en_contra','npxG+xAG_en_contra','Prog_con_Conduccion_en_contra','Prog_con_Pase_en_contra',
                          'Gls/90__en_contra','Ast/90__en_contra','G+A.1__en_contra','G-PK/90__en_contra','xG/90__en_contra','xAst/90__en_contra','xG+xAG/90__en_contra','npxG/90__en_contra',
                          'npxG+xAst/90__en_contra']
GoalKeeping_Own = ['Porteros_Participado','GA/90','SoTA','Saves','Save%','CS_a_favor','CS%_a_favor','PKsv','PKm','Penaltis_Save%']
GoalKeeping_Against = ['GA90','SoT','SoT_Detenidos_en_contra','Porc_SoT_Detenidos_en_contra','CS__en_contra','CS%__en_contra','PK_Detenidos_en_contra','PK_tirados_fuera']
GoalKeeping_Advance_Own = ['Goles_FK_en_contra','Goles_CK_en_contra','Goles_en_propia','PSxG','PSxG/SoT','PSxG+/-','PSxG_por_partido_en_contra','Pases_Completados_Portero',
                           'Pases_Largos_Intentados_Portero','Porc_Pases_Largos_Completados_Portero','Pases_Intentados_Portero','Centros_Interceptados_a_favor','Porc_Pases_Largos_Portero',
                           'Longitud_Media_Portero','Saques_Puerta','Porc_Saques_Puerta_Largo','Longitud_Media_Saque_Puerta','Centros_Recibidos','Centros_Detenidos_PT','Porc_Centros_Detenidos_PT',
                           'Acciones_Fuera_Area','Acciones_Fuera_Area_Por_Partido','Distancia_Media_PT_Porteria']
GoalKeeping_Advance_Against = ['Goles_FK_a_favor','Goles_CK_a_favor','Goles_en_propia_del_rival','PSxG_a_favor','PSxG/SoT_a_favor','PSxG+/-_a_favor','PSxG_por_partido_a_favor','Pases_Completados_Portero_rival',
                               'Pases_Largos_Intentados_Portero_rival','Porc_Pases_Largos_Completados_Portero_rival','Pases_Intentados_Portero_rival','Centros_Interceptados_en_contra',
                               'Porc_Pases_Largos_Portero_rival','Longitud_Media_Portero_rival','Saques_Puerta_rival','Porc_Saques_Puerta_Largo_rival','Longitud_Media_Saque_Puerta_rival','Centros_Colgados_a_favor',
                               'Centros_Detenidos_PT_rival','Porc_Centros_Detenidos_PT_rival','Acciones_Fuera_Area_PT_rival','Acciones_Fuera_Area_PT_rival_Por_Partido','Distancia_Media_PT_rival_Porteria']
Shooting_Own = ['Tiros_a_favor_por_partido','Tiros_a_puerta_a_favor_por_partido','Goles_por_disparo_a_favor','Goles_por_disparo_a_puerta','Distancia_media_tiros','Tiros_FK_a_favor',
                'Penaltis_Realizados_en_contra','PKatt','xG_sin_penaltis','xG_sin_penaltis_por_tiro','G-xG_a_favor','Goles_sin_penaltis_menos_xG_sin_penaltis']
Shooting_Against = ['Tiros_en_contra','Tiros_en_contra_por_partido','Tiros_a_puerta_en_contra_por_partido','Goles_por_disparo_en_contra','Goles_por_disparo_en_contra.1',
                    'Distancia_media_tiros_en_contra','Tiros_FK_en_contra','PK_y','Penaltis_intentados_en_contra','xG_en_contra','npxG_y','xG_en_contra_sin_penaltis_por_tiro',
                    'G-xG__en_contra','Goles_sin_penaltis_menos_xG_sin_penaltis_en_contra']
Passing_Own = ['Pases_Completados_a_favor','Pases_Intentados_a_favor','Porc_Pases_Completados_a_favor','Distancia_recorrida_pases_a_favor','Distancia_progresada_a_favor',
               'Pases_Cortos_Completados_a_favor','Pases_Cortos_Intentados_a_favor','Porc_Pases_Cortos_Completados_a_favor','Pases_Medios_Completados_a_favor','Pases_Medios_Intentados_a_favor',
               'Porc_Pases_Medios_Completados_a_favor','Pases_Largos_Completados_a_favor','Pases_Largos_Intentados_a_favor','Porc_Pases_Largos_Completados_a_favor','xAst_2_a_favor',
               'A-xAG_a_favor','Pases_Clave_a_favor','Pases_Primer_Tercio_a_favor','Pases_Area_Penalti_a_favor','Centros_Area_a_favor','Pases_Progresivos_a_favor']
Passing_Against = ['Pases_Completados_en_contra','Pases_Intentados_en_contra','Porc_Pases_Completados_en_contra','Distancia_recorrida_pases__en_contra','Distancia_progresada_en_contra',
                   'Pases_Cortos_Completados_en_contra','Pases_Cortos_Intentados_en_contra','Porc_Pases_Cortos_Completados_en_contra','Pases_Medios_Completados_en_contra','Pases_Medios_Intentados_en_contra',
                   'Porc_Pases_Medios_Completados_en_contra','Pases_Largos_Completados_en_contra','Pases_Largos_Intentados_en_contrar','Porc_Pases_Largos_Completados_en_contra','xAst_2_en_contra',
                   'A-xAG__en_contra','Pases_Clave_en_contra','Pases_Primer_Tercio_en_contra','Pases_Area_Penalti_en_contra','Centros_Area_en_contra','Pases_Progresivos_en_contra']
Pass_Types_Own = ['FK','Pases_Espalda_Defensa_a_favor','Cambios_Orientacion_a_favor','Centros_a_favor','Saques_Esquina_a_favor','Saques_Esquina_Interior_Area_a_favor',
                  'Saques_Esquina_Exterior_Area_a_favor','Saques_Esquina_Directo_Area_a_favor','Pases_a_favor_Fuera_Juego','Pases_Bloqueados_por_rival']
Pass_Types_Against = ['FK_en_contra','Pases_Espalda_Defensa_en_contra','Cambios_Orientacion_en_contra','Centros_en_contra','Saques_Esquina_en_contra','Saques_Esquina_Interior_Area_en_contra',
                      'Saques_Esquina_Exterior_Area_en_contra','Saques_Esquina_Directo_Area_en_contra','Pases_Fuera_Juego_en_contra','Pases_Bloqueados_por_equipo_propio']
Goal_and_Shot_Creation_Own = ['Accion_Tiro_Creada_a_favor','Accion_Tiro_Creada_por_Partido_a_favor','Regates_ante_Tiro_a_favor','Tiros_Generado_tras_Rebote_a_favor','Faltas_para_Disparar_a_favor',
                              'Acciones_Defensivas_Preceden_Tiro_a_favor','Acciones_Creadas_Gol_a_favor','Acciones_Creadas_Gol_por_partido_a_favor','Regates_ante_Gol_a_favor',
                              'Gol_Generado_tras_Rebote_a_favor','Goles_Falta_para_Disparar_a_favor','Acciones_Defensivas_Preceden_Gol_a_favor']
Goal_and_Shot_Creation_Against = ['Accion_Tiro_Creada_en_contra','Accion_Tiro_Creada_por_Partido_en_contra','Regates_ante_Tiro_en_contra','Tiros_Generado_tras_Rebote_en_contra','Faltas_para_Disparar_en_contra',
                                  'Acciones_Defensivas_Preceden_Tiro_en_contra','Acciones_Creadas_Gol_en_contra','Acciones_Creadas_Gol_por_partido_en_contra','Regates_ante_Gol_en_contra','Gol_Generado_tras_Rebote_en_contra',
                                  'Goles_Falta_para_Disparar_en_contra','Acciones_Defensivas_Preceden_Gol_en_contra']
Defensive_Actions_Own = ['Tackle_a_favor','Tackle_Ganado_a_favor','Tackle_1_Tercio_a_favor','Tackle_2_Tercio_a_favor','Tackle_3_Tercio_a_favor','Tackle_Regalte_a_favor_x','Tackle_Regate_Intentados_a_favor',
                         'Porc_Tackle_Ganado_a_favor','Bloqueos_a_favor','Tiros_Bloqueos_a_favor','Pases_Bloqueos_a_favor','Interceptaciones_a_favor','Tackles_Interceptaciones_a_favor',	'Despejes_a_favor','Errores_equipo_propio']
Defensive_Actions_Against = ['Tackle_en_contra','Tackle_Ganado_en_contra','Tackle_1_Tercio_en_contra','Tackle_2_Tercio_en_contra','Tackle_3_Tercio_en_contra','Tackle_Regalte_a_favor_y','Tackle_Regate_Intentados_en_contra',
                             'Porc_Tackle_Ganado_en_contra','Bloqueos_en_contra','Tiros_Bloqueos_en_contra','Pases_Bloqueos_en_contra','Interceptaciones_en_contra','Tackles_Interceptaciones_en_contra','Despejes_en_contra','Errores_equipos_rivales']
Possession_Own = ['Posesion_a_favor','Toques_a_favor','Toques_en_area_propia_a_favor','Toques_en_primer_tercio_a_favor','Toques_en_segundo_tercio_a_favor','Toques_en_tercer_tercio_a_favor','Toques_en_tarea_rival_a_favor',
                'Regate_Intentados_a_favor','Regates_Completados_a_favor','Porc_Regates_Completados_a_favor','Tackles_Regate_en_contra','Porcentaje_Tackles_Regate_en_contra','Conduccioness_a_favor','Distancia_Recorrida_por_Balon_a_favor',
                'Distancia_Recorrida_por_Balon_Adelante_a_favor','Toques_Balon_Adelante_a_favor','Conducciones_Ultimo_Tercio_a_favor','Conducciones_Area_Penalti_a_favor','Fallos_de_Control_equipo_propio','Perdida_Balon_euipo_propio_por_Tackle',
                'Pases_Recibidos_a_favor','Pases_Progresivos_Recibidos_a_favor']
Possession_Against = ['Posesion_en_contra','Toque_en_contra','Toques_en_area_rival_en_contra','Toques_en_primer_tercio_rival_en_contra','Toques_en_segundo_tercio_rival_en_contra','Toques_en_tercer_tercio_rival_en_contra','Toques_en_area_propia_en_contra',
                      'Regate_Intentados_en_contra','Regates_Completados_en_contra','Porc_Regates_Completados_en_contra','Tackles_Regate_a_favor','Porcentaje_Tackles_Regate_a_favor','Conduccioness_en_contra','Distancia_Recorrida_por_Balon_en_contra',
                      'Distancia_Recorrida_por_Balon_Adelante_en_contra','Toques_Balon_Adelante_en_contra','Conducciones_Ultimo_Tercio_en_contra','Conducciones_Area_Penalti_en_contra','Fallos_de_Control_equipo_rival','Perdida_Balon_euipo_rival_por_Tackle',
                      'Pases_Recibidos_en_contra','Pases_Progresivos_Recibidos_en_contra']
Miscellaneous_Own = ['2_Tarjetas_Amarillas_equipos_rivales','Faltas_cometidas_equipos_rivales','Fueras_Juego_equipos_rivales','Duelos_Aereos_ganados_equipos_rivales','Duelos_Aereos_perdidos_equipos_rivales','Porc_Duelos_Aereos_ganados_equipos_rivales']

# Creo los elegibles
lista_GRANDES_5_LIGAS_EUR_FBREF_202324_J28 = GRANDES_5_LIGAS_EUR_FBREF_202324_J28['Squad'].to_list()
distancias = ['euclidean','cosine']
Conjunto_Variables_dict = {
    'Clasificacion': Clasificacion,
    'Clasificacion_Home': Clasificacion_Home,
    'Clasificacion_Away': Clasificacion_Away,
    'Standard_Stats_Own': Standard_Stats_Own,
    'Standard_Stats_Against': Standard_Stats_Against,
    'GoalKeeping_Own': GoalKeeping_Own,
    'GoalKeeping_Against': GoalKeeping_Against,
    'GoalKeeping_Advance_Own': GoalKeeping_Advance_Own,
    'GoalKeeping_Advance_Against': GoalKeeping_Advance_Against,
    'Shooting_Own': Shooting_Own,
    'Shooting_Against': Shooting_Against,
    'Passing_Own': Passing_Own,
    'Passing_Against': Passing_Against,
    'Pass_Types_Own': Pass_Types_Own,
    'Pass_Types_Against': Pass_Types_Against,
    'Goal_and_Shot_Creation_Own': Goal_and_Shot_Creation_Own,
    'Goal_and_Shot_Creation_Against': Goal_and_Shot_Creation_Against,
    'Defensive_Actions_Own': Defensive_Actions_Own,
    'Defensive_Actions_Against': Defensive_Actions_Against,
    'Possession_Own': Possession_Own,
    'Possession_Against': Possession_Against,
    'Miscellaneous_Own': Miscellaneous_Own
}

# Intorduzco titulo del dashboard
st.write("<h1 style='text-align: center;'>GRAFICOS CUSTOMIZADOS</h1>", unsafe_allow_html=True)

# 3. Creacion de la funcion que ejecutará la app

# Funcion que me pinta ambos graficos
def pintar_graf_radar(Equipo, Variables, Distancia):
    # Me quedo con la temporada actual del equipo elegido
    df_Equipo = GRANDES_5_LIGAS_EUR_FBREF_202324_J28[GRANDES_5_LIGAS_EUR_FBREF_202324_J28['Squad']==Equipo]

    # PRIMER GRAFICO
    # Saco los minimos y maximos de cada variable y los vuelco en un dataframe
    # Crear un nuevo DataFrame vacío
    df_min_max = pd.DataFrame()

    # Iterar sobre las columnas del DataFrame original
    for col in CHAMP_GRANDES_5_LIGAS_EUR_FBREF_201617_202223.columns:
        if CHAMP_GRANDES_5_LIGAS_EUR_FBREF_201617_202223[col].dtype in ['int64', 'float64']:
            # Calcular el valor máximo y mínimo de la columna actual
            max_valor = max(CHAMP_GRANDES_5_LIGAS_EUR_FBREF_201617_202223[col])
            min_valor = min(CHAMP_GRANDES_5_LIGAS_EUR_FBREF_201617_202223[col])

            # Agregar el valor máximo y mínimo al nuevo DataFrame
            df_min_max[col+'_max'] = [max_valor]
            df_min_max[col+'_min'] = [min_valor]

    # Dividir el dataframe en dos filas
    df_max = df_min_max[df_min_max.columns[df_min_max.columns.str.endswith('_max')]].reset_index(drop=True)
    df_min = df_min_max[df_min_max.columns[df_min_max.columns.str.endswith('_min')]].reset_index(drop=True)

    # Renombrar las filas
    df_max.index = ['Maximo']
    df_min.index = ['Minimo']

    # Quitamos el sufijo '_min' o '_max'
    df_max.columns = df_max.columns.str.replace('_min', '').str.replace('_max', '')
    df_min.columns = df_min.columns.str.replace('_min', '').str.replace('_max', '')

    df_min_max = pd.concat([df_min, df_max], axis=0)

    df_Equipo.set_index('Squad', inplace=True)
    df_min_max_equipo = pd.concat([df_min_max,df_Equipo.iloc[:,2:]])

    # Definir los datos
    labels = np.array(Variables)
    min_stats = np.array(df_min_max_equipo[Variables].iloc[0]) # Valores mínimos para cada categoría
    max_stats = np.array(df_min_max_equipo[Variables].iloc[1]) # Valores máximos para cada categoría
    team_stats = np.array(df_min_max_equipo[Variables].iloc[2]) # Valores máximos para cada categoría

    # Número de variables
    num_vars = len(labels)

    # Ángulos para cada eje
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # El gráfico es un círculo, entonces cerramos el ciclo
    min_stats = np.concatenate((min_stats,[min_stats[0]]))
    max_stats = np.concatenate((max_stats,[max_stats[0]]))
    team_stats = np.concatenate((team_stats,[team_stats[0]]))
    angles += angles[:1]

    # Crear la figura y el eje polar
    fig1, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    # Dibujar el radar para los valores mínimos
    ax.fill(angles, min_stats, color='red', alpha=0.25, label='Valores mínimos')
    # Dibujar el radar para los valores máximos
    ax.fill(angles, max_stats, color='blue', alpha=0.25, label='Valores máximos')
    # Dibujar el radar para los valores máximos
    ax.fill(angles, team_stats, color='yellow', alpha=0.25, label=Equipo)

    # Establecer las etiquetas de las variables en el eje polar
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Añadir leyenda
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

    # Para que la leyenda entre en los margenes de la imagen al descargarla
    plt.tight_layout()

    # Ponemos la opcion de guardar el grafico como imagen
    bufferq = io.BytesIO()
    fig1.savefig(bufferq, format='jpeg')
    bufferq.seek(0)

    # Mostrar el gráfico
    columna1.pyplot(fig1)

    # Boton de descargar
    st.download_button(label="Descargar gráfico 1 como imagen",
                       data=bufferq.getvalue(),
                       file_name='Grafico_1.jpeg',
                       mime="image/jpeg")

    # SEGUNDO GRAFICO
    # Filtrar el dataframe para obtener solo las filas correspondientes al equipo dado
    team_data = df_Equipo[['Temporada'] + Variables]

    # Filtrar el dataframe para obtener solo las filas correspondientes a los otros equipos
    other_teams_data = CHAMP_GRANDES_5_LIGAS_EUR_FBREF_201617_202223[['Squad','Temporada'] + Variables]

    if Distancia == 'cosine':
        # Calcular la distancia del coseno para cada equipo
        distances = cosine_similarity(other_teams_data[Variables].values, team_data[Variables].values)
        distances = distances[:, 0]  # Seleccionamos la primera columna (similitud con el equipo dado)
    elif Distancia == 'euclidean':
        # Calcular la distancia euclidiana para cada equipo
        distances = np.sqrt(np.sum((other_teams_data[Variables].values - team_data[Variables].values)**2, axis=1))
    else:
        raise ValueError("Tipo de medida de distancia no válido. Utiliza 'cosine' o 'euclidean'.")

    # Crear un nuevo dataframe con los equipos, temporadas y sus distancias
    result_df = pd.DataFrame({'Temporada': other_teams_data['Temporada'], 'Equipo': CHAMP_GRANDES_5_LIGAS_EUR_FBREF_201617_202223['Squad'], 'Distancia': distances})

    # Ordenar los resultados en función de la distancia (menor similitud a mayor similitud)
    result_df = result_df.sort_values(by='Distancia', ascending=True)

    # Guardar el primer registro en un nuevo dataframe llamado df_similar_team
    df_similar_team = result_df.head(1)

    Equipo_Similar = df_similar_team['Equipo'].iloc[0]
    Temporada_Similar = df_similar_team['Temporada'].iloc[0]

    df_similar_team = GRANDES_5_LIGAS_EUR_FBREF_201617_202223[(GRANDES_5_LIGAS_EUR_FBREF_201617_202223['Squad']==Equipo_Similar)&(GRANDES_5_LIGAS_EUR_FBREF_201617_202223['Temporada']==Temporada_Similar)]
    df_similar_team.set_index('Squad', inplace=True)
    df_equipo_similar = pd.concat([df_Equipo,df_similar_team.iloc[:,2:]])

    # Definir los datos
    labels = np.array(Variables)
    team_stats = np.array(df_equipo_similar[Variables].iloc[0]) # Valores equipo de estudio para cada categoría
    similar_team_stats = np.array(df_equipo_similar[Variables].iloc[1]) # Valores equipo de estudio para cada categoría

    # Número de variables
    num_vars = len(labels)

    # Ángulos para cada eje
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # El gráfico es un círculo, entonces cerramos el ciclo
    team_stats = np.concatenate((team_stats,[team_stats[0]]))
    similar_team_stats = np.concatenate((similar_team_stats,[similar_team_stats[0]]))
    angles += angles[:1]

    # Crear la figura y el eje polar
    fig2, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    # Dibujar el radar para los valores del equipo de estudio
    ax.fill(angles, team_stats, color='yellow', alpha=0.25, label=Equipo)
    # Dibujar el radar para los valores del equipo campeon mas similar
    ax.fill(angles, similar_team_stats, color='green', alpha=0.25, label=Equipo_Similar)

    # Establecer las etiquetas de las variables en el eje polar
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Añadir leyenda
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    # Para que la leyenda entre en los margenes de la imagen al descargarla
    plt.tight_layout()

    # Ponemos la opcion de guardar el grafico como imagen
    buffer2 = io.BytesIO()
    fig2.savefig(buffer2, format='jpg')
    buffer2.seek(0)

    # Mostrar el gráfico
    columna2.pyplot(fig2)
    columna2.markdown('El equipo mas similar es el {} de la temporada {}'.format(Equipo_Similar, Temporada_Similar))

    # Boton de descargar
    st.download_button(label="Descargar gráfico 2 como imagen",
                       data=buffer2.getvalue(),
                       file_name='Grafico_2.jpg',
                       mime="image/jpg")

columna1, columna2 = st.columns(2)

# 4. Widgets de la app

# Titulo de la barra lateral de la app
st.sidebar.header("SELECTOR")

# Crear widgets para seleccionar los parámetros
equipo_seleccionado = st.sidebar.selectbox('Selecciona un equipo:', lista_GRANDES_5_LIGAS_EUR_FBREF_202324_J28)
variables_seleccionadas = st.sidebar.selectbox('Selecciona las variables:',  Conjunto_Variables_dict.keys())
distancia_seleccionada = st.sidebar.selectbox('Selecciona la distancia:', distancias)

# Llamar a la función con los parámetros seleccionados
if st.sidebar.button('Generar gráficos'):
    pintar_graf_radar(equipo_seleccionado, Conjunto_Variables_dict[variables_seleccionadas], distancia_seleccionada)