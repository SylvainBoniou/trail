import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
from streamlit_option_menu import option_menu
from datetime import datetime, date, timedelta
import pycountry_convert as pc
from tqdm import tqdm

st.set_page_config(
    page_title="DataTrail",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# On charge les données
df = pd.read_csv('./data/utmb-race-data-sheet.csv', sep = ',')

with st.sidebar:
  selected = option_menu(
    menu_title = "Menu",
    options = ["Course","Coureurs"],
    icons = ["search","speedometer2"],
    menu_icon = "cast",
    default_index = 0,
    styles={
          "container": {"padding": "0!important", "background-color": "#fafafa"},
          "icon": {"color": "orange", "font-size": "25px"},
          "nav-link": { "font-family": "Oxanium, sans-serif","font-size": "700", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
          "nav-link-selected": {"background-color": "rgb(0, 13, 68)"},
    }
  )

t1, t2 = st.columns((0.2,0.8))

# t1.image('./data/utmb.png', width = 120)
t2.title('Application de la data dans le trail')

def make_donut(input_response, input_text, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100 - input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            # range=['#29b5e8', '#155F7A']),  # 31333F
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)

    text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700,
                          fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            range=chart_color),  # 31333F
                        legend=None),
    ).properties(width=130, height=130)
    return plot_bg + plot + text

def float_to_hhmm(hours_float):
    hours = int(hours_float)  # Partie entière (heures)
    minutes = int((hours_float - hours) * 60)  # Partie décimale convertie en minutes
    return f"{hours:02}h{minutes:02}min"

if selected == "Course":
    t1, t2 = st.columns((0.1, 1))
    t1.image('./data/utmb.png', width = 120)
    t2.header('Recherche de courses UTMB World Series',divider="gray")

    col1, col2, col3 = st.columns(3)

    with col1:
        df.sort_values('Country', ascending=True, inplace=True)
        liste_pays = df['Country'].unique()
        serie = pd.Series(liste_pays)
        df['NomCountry'] = df['Country']

        ma_liste_sans_nan = serie.dropna().tolist()

        for i in range(126):
            if ma_liste_sans_nan[i]!='XK':
                df['NomCountry'] = df['NomCountry'].replace(ma_liste_sans_nan[i], pc.country_alpha2_to_country_name(ma_liste_sans_nan[i]))
                ma_liste_sans_nan[i] = pc.country_alpha2_to_country_name(ma_liste_sans_nan[i])

        pays = st.selectbox('Choisissez un pays', ma_liste_sans_nan, index=None, placeholder="Sélectionnez un pays...")

    with col2:
        today = datetime.now()
        next_year = today.year
        jan_1 = date(2014, 1, 1)
        dec_31 = date(next_year, 12, 31)

        d = st.date_input(
            "Choisissez une date",
            (jan_1, date(next_year, 12, 31)),
            jan_1,
            dec_31,
            format="DD/MM/YYYY",
        )

    with col3:
        distance_min, distance_max = st.select_slider('Choisissez une distance', options=[0, 10, 20 , 30, 40, 50, 60, 70, 80, 90, 100, 110,120,130,140,150,160,170,180,190,200,210,220,230,240,250], value=(0,250))

    col4, col5, col6 = st.columns(3)
    with col4:
        denivele_min, denivele_max = st.select_slider('Choisissez un dénivelé', options=[0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000,12500,13000,13500,14000,14500,15000], value=(0,15000))

    with col5:
        df.sort_values('Race Category', ascending=True, inplace=True)
        liste_categorie = df['Race Category'].unique()
        categorie = st.selectbox('Choisissez une catégorie', liste_categorie, index=None,
                                 placeholder="Sélectionnez une catégorie...")

    st.divider()

    df['Date'] =  pd.to_datetime(df['Year'].astype(str) + '-01-01') + pd.to_timedelta(df['Day'], unit='D')
    df['Date'] = df['Date'].dt.date
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    if pays:
        df = df[df['NomCountry'] == pays]

    if categorie:
        df = df[df['Race Category'] == categorie]

    if denivele_min:
        df = df[(df['Elevation Gain'] > denivele_min) & (df['Elevation Gain'] < denivele_max)]

    if distance_min:
        df = df[(df['Distance'] > distance_min) & (df['Distance'] < distance_max)]

    if d:
        datedebut= datetime.combine(d[0], datetime.min.time())
        datefin = datetime.combine(d[1], datetime.min.time())
        # datetime.strptime(d[0], '%d/%m/%Y')
        df = df[(df['Date'] > datedebut) & (df['Date'] < datefin)]
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d/%m/%Y')


    colonne = ['Year','Race Title','Distance','Elevation Gain',  'Date', 'Country', 'Race Category','Winning Time','Last Time','Mean Finish Time','N Participants','N DNF', 'N Women']
    df = df[colonne]

    df = df.rename(columns={'Year': 'Année', 'Race Title': 'Course', 'N Participants': 'Nb participants', 'Race Category': 'Catégorie','Country': 'Pays', 'Distance': 'Distance (km)','Elevation Gain': 'Dénivelé (m)',  'Winning Time': 'Meilleur temps', 'Mean Finish Time': 'Temps moyen','Last Time': 'Temps le + long', 'N DNF': '% DNF', 'N Women': '% Femmes'})

    df['Meilleur temps'] =  df['Meilleur temps'].apply(lambda x: float_to_hhmm(x) if pd.notna(x) else None)
    df['Temps moyen'] =  df['Temps moyen'].apply(lambda x: float_to_hhmm(x) if pd.notna(x) else None)
    df['Temps le + long'] =  df['Temps le + long'].apply(lambda x: float_to_hhmm(x) if pd.notna(x) else None)
    df['% DNF']= df['% DNF']*100/df['Nb participants']
    df['% DNF'] = df['% DNF'].round(2).astype(str)+'%'
    df['% Femmes'] = df['% Femmes'] * 100 / df['Nb participants']
    df['% Femmes'] = df['% Femmes'].round(2).astype(str) + '%'
    df['Pays'] ="https://cdn.jsdelivr.net/gh/lipis/flag-icons@6.6.6/flags/4x3/"+df['Pays'].str.lower()+".svg"
    df['Catégorie'] = "https://res.cloudinary.com/utmb-world/image/upload/q_auto/f_auto/c_fill,g_auto/if_w_gt_1920/c_scale,w_1920/if_end/v1/Common/categories/"+df['Catégorie']

    st.dataframe(
        df,
        column_config={
            "Pays" : st.column_config.ImageColumn("Pays" ),
            "Catégorie" : st.column_config.ImageColumn("Catégorie", width="small" ),
            "Année": st.column_config.NumberColumn(
                format="%d",
                # format="%d ⭐",
            ),
            "Dénivelé (m)": st.column_config.NumberColumn(
                format="%d",
            ),
            "% DNF": st.column_config.ProgressColumn(
                "% DNF",
                min_value=0,
                max_value=0.9,
            ),
            "% Femmes": st.column_config.ProgressColumn(
                "% Femmes",
                min_value=0,
                max_value=0.9,
                ),
        },
        hide_index=True,
    )

    # st.divider()
    # st.map(df, latitude="Latitude", longitude="Longitude", size="Nb participants")
    # df['NB']=df['Nb participants']*3
    # point_layer = pydeck.Layer(
    #     "ScatterplotLayer",
    #     data=df,
    #     id="capital-cities",
    #     get_position=["Longitude", "Latitude"],
    #     get_color="[255, 75, 75]",
    #     pickable=True,
    #     auto_highlight=True,
    #     get_radius="NB",
    # )
    #
    # view_state = pydeck.ViewState(
    #     latitude=40, longitude=-117, controller=True, zoom=2.4, pitch=30
    # )
    #
    # chart = pydeck.Deck(
    #     point_layer,
    #     initial_view_state=view_state,
    #     tooltip={"text": "{Course}\nNb Participants: {Nb participants}"},
    # )
    #
    # event = st.pydeck_chart(chart)
    #
    # event.selection

if selected == "Coureurs":
    t1, t2 = st.columns((0.1, 1))
    t1.image('https://res.cloudinary.com/hy4kyit2a/f_auto,fl_lossy,q_70/learn/modules/data-analytics-fundamentals/4f5976bff8a0a632f73fd65542fa3b6f_badge.png', width = 120)
    t2.header('Estimation de performance',divider="gray")

    # On charge les données
    df_coureurs = pd.read_csv('./data/resultats_utmb.csv', sep = ',')

    col1, col2, col3 = st.columns(3)

    df_coureurs = df_coureurs[df_coureurs['Index'] != '-']
    df_coureurs = df_coureurs.dropna(subset=['Index'])

    with col1:
        df_coureurs.sort_values('Index', ascending=True, inplace=True)
        # liste_index = df_coureurs['Index'].unique()
        liste_index ={1:'< 400',2:'400-425',3:'425-450',4:'450-475',5:'475-500',6:'500-525',7:'525-550',8:'550-575',9:'575-600',10:'600-625',11:'625-650',12:'650-675',13:'675-700',14:'700-725',15:'725-750',16:'750-775',17:'775-800',18:'800-825',19:'825-850',20:'850-875',21:'875-900',22:'900-925',23:'925-950',24:'> 950'}
        liste_index ={2:'400-425',3:'425-450',4:'450-475',5:'475-500',6:'500-525',7:'525-550',8:'550-575',9:'575-600',10:'600-625',11:'625-650',12:'650-675',13:'675-700',14:'700-725',15:'725-750',16:'750-775',17:'775-800',18:'800-825',19:'825-850',20:'850-875',21:'875-900',22:'900-925',23:'925-950'}

        index = st.selectbox('Choisissez votre catégorie d''index UTMB', liste_index.values(), index=None, placeholder="Sélectionnez votre catégorie d''index UTMB...")

    with col2:
        course = st.radio(
            "Choisissez votre course 👇",
            ["UTMB", "Diagonale des fous"],
            index=1,
            horizontal  = True
        )

    if course!='UTMB':
        df_coureurs = pd.read_csv('./data/resultats_diag.csv', sep=',')
        df_coureurs = df_coureurs[(df_coureurs['Index'] != 0)]

        df_coureurs = df_coureurs[(df_coureurs['Classement'] != 'DNF')]
        df_coureurs['Classement'] = df_coureurs['Classement'].astype(int)

    df_coureurs = df_coureurs[df_coureurs['Index'] != 'NaN']
    df_coureurs['Index'] = df_coureurs['Index'].astype(int)

    if index:
        if index=="< 400":
          df_coureurs = df_coureurs[(df_coureurs['Index'] < 400)]
        else:
            if index=="> 950":
                df_coureurs = df_coureurs[(df_coureurs['Index'] > 950)]
            else:
                df_coureurs = df_coureurs[(df_coureurs['Index'] > int(index[:3])) & (df_coureurs['Index'] < int(index[4:]))]

    if df_coureurs.empty:
        st.error('Ne faites pas cette ccourse vous n''avez pas le niveau !!!', icon="🚨")
        st.image("https://storage.googleapis.com/utrailbucket/2024/08/Capture-decran-2024-08-31-a-01.51.19-750x536.jpg")
    else:
        temps_max = pd.to_timedelta(df_coureurs['Temps']).max()
        heures_max, remainder_max = divmod(temps_max.total_seconds(), 3600)
        minutes_max, secondes_max = divmod(remainder_max, 60)
        temps_max_hhmmss = f"{int(heures_max):02}:{int(minutes_max):02}:{int(secondes_max):02}"

        temps_min = pd.to_timedelta(df_coureurs['Temps']).min()
        heures_min, remainder_min = divmod(temps_min.total_seconds(), 3600)
        minutes_min, secondes_min = divmod(remainder_min, 60)
        temps_min_hhmmss = f"{int(heures_min):02}:{int(minutes_min):02}:{int(secondes_min):02}"

        temps_moy = pd.to_timedelta(df_coureurs['Temps']).mean()
        heures_moy, remainder_moy = divmod(temps_moy.total_seconds(), 3600)
        minutes_moy, secondes_moy = divmod(remainder_moy, 60)
        temps_moy_hhmmss = f"{int(heures_moy):02}:{int(minutes_moy):02}:{int(secondes_moy):02}"

        classement_max = df_coureurs['Classement'].max()
        classement_min = df_coureurs['Classement'].min()
        classement_moyen = round((df_coureurs['Classement'].mean()))

        data = {
            'Course': 'UTMB',
            'Meilleur classement 🥇': classement_min,
            'Classement moyen': classement_moyen,
            'Classement le + pourri 🚷': classement_max,
            'Meilleur temps 🤩': temps_min_hhmmss,
            'Temps moyen': temps_moy_hhmmss,
            'Temps le + pourri ❤️‍🩹': temps_max_hhmmss
        }

        df_final = pd.DataFrame(data, index=[1])

        st.dataframe(
            df_final,
            column_config={
                "Classement moyen": st.column_config.NumberColumn(
                    format="%d",
                ),
                "Classement le + pourri 🚷": st.column_config.NumberColumn(
                    format="%d",
                )
            },
            hide_index=True,
        )

        st.markdown("""
        <style>
        .big-font {
            font-size:12px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="big-font">*Résulats basés sur les 9 dernières années pour l''UTMB et les 6 dernières années pour la diagonale des fous.</p>', unsafe_allow_html=True)

        fig = px.scatter(df_coureurs, x="Index", y="Temps",
                            hover_name='Nom',
                            title="Temps réalisés selon l'index UTMB")
        fig.update_layout(
            margin=dict(l=0, r=0, t=25, b=0),
            height=600,
        )
        fig.update_layout(yaxis=dict(type='category', categoryorder='category ascending'))

        st.plotly_chart(fig, use_container_width=True)

        # test =  df_coureurs['Classement'].count()
        # fig2 = px.bar(df_coureurs, x="Index", y="Classement")
        # st.plotly_chart(fig2, use_container_width=True)
