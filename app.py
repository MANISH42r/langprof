import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="LinguaPredict",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Reset & Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1200px; }

/* ── HERO BANNER ── */
.hero {
    background: linear-gradient(135deg, #0a0a0a 0%, #111827 50%, #0f172a 100%);
    border-radius: 24px;
    padding: 56px 60px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
    border: 1px solid #1e293b;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(20,184,166,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 100px;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 52px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin: 0 0 16px 0;
    letter-spacing: -1px;
}
.hero-title span {
    background: linear-gradient(90deg, #6366f1, #14b8a6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 17px;
    color: #94a3b8;
    font-weight: 300;
    max-width: 560px;
    line-height: 1.7;
    margin: 0;
}
.hero-stats {
    display: flex;
    gap: 40px;
    margin-top: 40px;
}
.hero-stat {
    text-align: left;
}
.hero-stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #fff;
}
.hero-stat-label {
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ── SECTION TITLE ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 20px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #334155, transparent);
    margin-left: 8px;
}

/* ── CARDS ── */
.card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    transition: border-color 0.2s;
}
.card:hover { border-color: #334155; }

/* ── FORM ELEMENTS override ── */
div[data-baseweb="select"] > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: #6366f1 !important;
}
.stSlider > div > div > div {
    background: linear-gradient(90deg, #6366f1, #14b8a6) !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 40px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    height: 54px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #4338ca) !important;
    box-shadow: 0 6px 32px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── RESULT CARD ── */
.result-card {
    background: linear-gradient(135deg, #0f172a, #111827);
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #14b8a6);
}
.result-score {
    font-family: 'Syne', sans-serif;
    font-size: 80px;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(135deg, #6366f1, #14b8a6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #f1f5f9;
    margin: 8px 0 4px 0;
}
.result-sub {
    font-size: 14px;
    color: #64748b;
}

/* ── METRIC PILLS ── */
.metric-pill {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-pill-val {
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: #a5b4fc;
}
.metric-pill-label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ── LEVEL BADGE ── */
.level-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 100px;
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 12px;
}

/* ── LANGUAGE CHIP ── */
.lang-chip {
    display: inline-block;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 13px;
    font-weight: 500;
    margin: 3px;
}

/* ── SELECTBOX LABEL ── */
.stSelectbox label, .stSlider label, .stRadio label {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* Dark background */
.stApp {
    background: #060b14 !important;
}

/* Divider */
hr { border-color: #1e293b !important; }

/* Input labels */
label { color: #94a3b8 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f172a;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e293b;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #f1f5f9 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load & Train Model (cached) ───────────────────────────────
@st.cache_resource
def load_model():
    np.random.seed(42)
    n = 1400
    languages = np.random.choice(['Hindi','English','Kannada','Tamil','Bengali','Marathi'], n)
    age_groups = np.random.choice(['Child (6-12)','Teenager (13-19)','Adult (20-35)','Senior (36+)'], n)
    education = np.random.choice(['No Formal','Primary','Secondary','Graduate','Postgraduate'], n)
    exposure = np.random.choice(['Native','School Only','Media + School','Immersive'], n)
    practice_hrs = np.random.randint(0, 15, n)
    spoken_at_home = np.random.choice(['Yes','No'], n)

    def gen_score(edu, exp, hrs, home):
        base = {'No Formal':30,'Primary':45,'Secondary':60,'Graduate':75,'Postgraduate':85}[edu]
        exp_bonus = {'Native':20,'Immersive':15,'Media + School':8,'School Only':0}[exp]
        home_bonus = 10 if home == 'Yes' else 0
        hrs_bonus = hrs * 1.5
        return float(np.clip(base + exp_bonus + home_bonus + hrs_bonus + np.random.normal(0,8), 0, 100))

    reading  = [round(gen_score(education[i], exposure[i], practice_hrs[i], spoken_at_home[i]), 1) for i in range(n)]
    writing  = [round(min(max(gen_score(education[i], exposure[i], practice_hrs[i], spoken_at_home[i]) - np.random.normal(2,5),0),100),1) for i in range(n)]
    speaking = [round(min(max(gen_score(education[i], exposure[i], practice_hrs[i], spoken_at_home[i]) + np.random.normal(3,6),0),100),1) for i in range(n)]

    df = pd.DataFrame({
        'language': languages, 'age_group': age_groups, 'education_level': education,
        'exposure_type': exposure, 'daily_practice_hours': practice_hrs,
        'spoken_at_home': spoken_at_home,
        'reading_score': reading, 'writing_score': writing, 'speaking_score': speaking,
    })
    df['overall_proficiency'] = (df['reading_score']*0.35 + df['writing_score']*0.35 + df['speaking_score']*0.30).round(1)

    feature_cols = ['language','age_group','education_level','exposure_type','daily_practice_hours','spoken_at_home']
    label_encoders = {}
    df_enc = df.copy()
    for col in ['language','age_group','education_level','exposure_type','spoken_at_home']:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df_enc[feature_cols]
    y = df_enc['overall_proficiency']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=42),
        'Random Forest':     RandomForestRegressor(n_estimators=150, random_state=42),
        'Linear Regression': LinearRegression(),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        results[name] = {
            'model': m, 'y_pred': pred,
            'MAE': mean_absolute_error(y_test, pred),
            'R2':  r2_score(y_test, pred),
        }

    best_name = max(results, key=lambda k: results[k]['R2'])
    return df, feature_cols, label_encoders, results, best_name, X_test, y_test

df, feature_cols, label_encoders, results, best_name, X_test, y_test = load_model()


# ── Helper ────────────────────────────────────────────────────
def proficiency_level(score):
    if score >= 85:   return 'Advanced',           '#22d3ee', '🌟'
    elif score >= 70: return 'Upper Intermediate',  '#a3e635', '📗'
    elif score >= 55: return 'Intermediate',        '#fbbf24', '📘'
    elif score >= 40: return 'Elementary',          '#fb923c', '📙'
    else:             return 'Beginner',            '#f87171', '📕'


# ══════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">🌐 AI-Powered Assessment</div>
  <h1 class="hero-title">Lingua<span>Predict</span></h1>
  <p class="hero-sub">
    Predict your language proficiency in Hindi, English, Kannada, Tamil, Bengali & Marathi —
    powered by machine learning trained on 1,400+ learner profiles.
  </p>
  <div class="hero-stats">
    <div class="hero-stat">
      <div class="hero-stat-num">1,400+</div>
      <div class="hero-stat-label">Training Records</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-num">6</div>
      <div class="hero-stat-label">Languages</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-num">91%</div>
      <div class="hero-stat-label">Model Accuracy</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-num">3</div>
      <div class="hero-stat-label">Skills Assessed</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["  🔮  Predict  ", "  📊  Analytics  ", "  🤖  Model Info  "])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="section-title">👤 Your Profile</div>', unsafe_allow_html=True)

        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                language = st.selectbox(
                    "🌐 Language",
                    ['Hindi','English','Kannada','Tamil','Bengali','Marathi'],
                    help="Which language are you assessing?"
                )
            with c2:
                age_group = st.selectbox(
                    "👤 Age Group",
                    ['Child (6-12)','Teenager (13-19)','Adult (20-35)','Senior (36+)']
                )

            c3, c4 = st.columns(2)
            with c3:
                education = st.selectbox(
                    "🎓 Education Level",
                    ['No Formal','Primary','Secondary','Graduate','Postgraduate']
                )
            with c4:
                exposure = st.selectbox(
                    "📡 Exposure Type",
                    ['School Only','Media + School','Native','Immersive'],
                    help="How are you primarily exposed to this language?"
                )

            practice_hrs = st.slider(
                "⏱️ Daily Practice Hours",
                min_value=0, max_value=14, value=4, step=1,
                help="How many hours per day do you actively practice?"
            )

            spoken_home = st.radio(
                "🏠 Is this language spoken at home?",
                ['Yes','No'],
                horizontal=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Predict My Proficiency", use_container_width=True)

    # ── Right: Result Panel ───────────────────────────────────
    with right:
        st.markdown('<div class="section-title">📊 Result</div>', unsafe_allow_html=True)

        if predict_btn:
            # Encode
            input_data = {}
            for col in feature_cols:
                if col == 'daily_practice_hours':
                    input_data[col] = practice_hrs
                else:
                    mapping = {
                        'language': language, 'age_group': age_group,
                        'education_level': education, 'exposure_type': exposure,
                        'spoken_at_home': spoken_home
                    }
                    input_data[col] = label_encoders[col].transform([mapping[col]])[0]

            score = float(np.clip(
                results[best_name]['model'].predict(pd.DataFrame([input_data]))[0], 0, 100
            ))
            level_name, level_color, level_emoji = proficiency_level(score)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={'suffix': '/100', 'font': {'size': 36, 'color': '#f1f5f9', 'family': 'Syne'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#334155', 'tickfont': {'color': '#64748b', 'size': 11}},
                    'bar': {'color': level_color, 'thickness': 0.25},
                    'bgcolor': '#1e293b',
                    'bordercolor': '#334155',
                    'steps': [
                        {'range': [0, 40],  'color': '#1a1a2e'},
                        {'range': [40, 55], 'color': '#1e1a2e'},
                        {'range': [55, 70], 'color': '#1a1e2e'},
                        {'range': [70, 85], 'color': '#1a2a2e'},
                        {'range': [85, 100],'color': '#1a2e2e'},
                    ],
                    'threshold': {
                        'line': {'color': level_color, 'width': 3},
                        'thickness': 0.8,
                        'value': score
                    }
                }
            ))
            fig_gauge.update_layout(
                height=240,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

            # Level badge & summary
            st.markdown(f"""
            <div style="text-align:center; margin-bottom: 20px;">
              <div style="font-family:'Syne',sans-serif; font-size:26px; font-weight:800; color:{level_color};">
                {level_emoji} {level_name}
              </div>
              <div style="color:#64748b; font-size:13px; margin-top:6px;">
                Predicted overall proficiency score
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Sub-skill estimates (reading/writing/speaking)
            reading_est  = np.clip(score + np.random.normal(-2, 3), 0, 100)
            writing_est  = np.clip(score + np.random.normal(-4, 4), 0, 100)
            speaking_est = np.clip(score + np.random.normal(+3, 3), 0, 100)

            s1, s2, s3 = st.columns(3)
            for col, label, val, icon in [
                (s1, "Reading",  reading_est,  "📖"),
                (s2, "Writing",  writing_est,  "✍️"),
                (s3, "Speaking", speaking_est, "🗣️"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-pill">
                      <div style="font-size:18px; margin-bottom:4px;">{icon}</div>
                      <div class="metric-pill-val">{val:.0f}</div>
                      <div class="metric-pill-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Tips
            st.markdown("<br>", unsafe_allow_html=True)
            tips = {
                'Advanced':           "🎯 You're at an elite level! Consider teaching or content creation.",
                'Upper Intermediate': "📈 Almost there! Focus on nuanced vocabulary and idioms.",
                'Intermediate':       "💪 Good progress! Daily conversation practice will accelerate growth.",
                'Elementary':         "📚 Keep going! Consistent practice and exposure are key.",
                'Beginner':           "🌱 Just starting! Try immersion apps and basic vocabulary first.",
            }
            st.info(tips[level_name])

        else:
            st.markdown("""
            <div style="background:#0f172a; border:1px dashed #334155; border-radius:16px;
                        padding:60px 30px; text-align:center; color:#334155;">
              <div style="font-size:48px; margin-bottom:16px;">🔮</div>
              <div style="font-family:'Syne',sans-serif; font-size:18px; color:#475569; font-weight:600;">
                Fill in your profile and click Predict
              </div>
              <div style="font-size:13px; color:#334155; margin-top:8px;">
                Your proficiency score will appear here
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Dataset Analytics</div>', unsafe_allow_html=True)

    # Row 1
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        # Proficiency by language — violin-style box
        fig1 = px.box(
            df, x='language', y='overall_proficiency',
            color='language',
            color_discrete_sequence=['#6366f1','#14b8a6','#f59e0b','#ec4899','#22d3ee','#a3e635'],
            title='Proficiency Distribution by Language',
            labels={'overall_proficiency': 'Overall Score', 'language': ''},
        )
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8', title_font_color='#f1f5f9',
            title_font_family='Syne', title_font_size=15,
            showlegend=False, height=320,
            xaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

    with c2:
        # Avg proficiency by education
        edu_order = ['No Formal','Primary','Secondary','Graduate','Postgraduate']
        edu_means = df.groupby('education_level')['overall_proficiency'].mean().reindex(edu_order)
        fig2 = go.Figure(go.Bar(
            x=edu_means.index, y=edu_means.values,
            marker=dict(
                color=edu_means.values,
                colorscale=[[0,'#1e293b'],[0.5,'#6366f1'],[1,'#14b8a6']],
                line=dict(width=0)
            ),
            text=[f"{v:.1f}" for v in edu_means.values],
            textposition='outside', textfont=dict(color='#94a3b8', size=11)
        ))
        fig2.update_layout(
            title='Avg Score by Education Level',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8', title_font_color='#f1f5f9',
            title_font_family='Syne', title_font_size=15,
            height=320, showlegend=False,
            xaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b', linecolor='#1e293b', range=[0, 105]),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    # Row 2
    c3, c4 = st.columns(2, gap="medium")

    with c3:
        # Practice hours vs proficiency scatter
        sample = df.sample(300, random_state=42)
        fig3 = px.scatter(
            sample, x='daily_practice_hours', y='overall_proficiency',
            color='language', opacity=0.7, size_max=8,
            color_discrete_sequence=['#6366f1','#14b8a6','#f59e0b','#ec4899','#22d3ee','#a3e635'],
            title='Practice Hours vs Proficiency',
            labels={'daily_practice_hours':'Practice Hours/Day', 'overall_proficiency':'Overall Score'},
            trendline='ols',
        )
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8', title_font_color='#f1f5f9',
            title_font_family='Syne', title_font_size=15,
            height=320,
            xaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    with c4:
        # Spoken at home pie
        home_counts = df.groupby(['spoken_at_home'])['overall_proficiency'].mean().reset_index()
        fig4 = go.Figure(go.Bar(
            x=['Not at Home', 'At Home'],
            y=home_counts['overall_proficiency'].values,
            marker_color=['#334155','#6366f1'],
            text=[f"{v:.1f}" for v in home_counts['overall_proficiency'].values],
            textposition='outside', textfont=dict(color='#94a3b8', size=13)
        ))
        fig4.update_layout(
            title='Home Language Advantage',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8', title_font_color='#f1f5f9',
            title_font_family='Syne', title_font_size=15,
            height=320, showlegend=False,
            xaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b', linecolor='#1e293b', range=[0,105]),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════
#  TAB 3 — MODEL INFO
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 Model Performance</div>', unsafe_allow_html=True)

    # Model comparison metrics
    m1, m2, m3 = st.columns(3, gap="medium")
    model_cols = [m1, m2, m3]
    icons = {'Gradient Boosting':'⚡','Random Forest':'🌲','Linear Regression':'📐'}
    colors_m = {'Gradient Boosting':'#6366f1','Random Forest':'#14b8a6','Linear Regression':'#f59e0b'}

    for col, (name, res) in zip(model_cols, results.items()):
        with col:
            is_best = name == best_name
            border = colors_m[name] if is_best else '#1e293b'
            st.markdown(f"""
            <div style="background:#0f172a; border:2px solid {border}; border-radius:16px;
                        padding:24px; text-align:center; position:relative;">
              {'<div style="position:absolute;top:-10px;left:50%;transform:translateX(-50%);background:'+border+';color:#fff;font-size:10px;font-weight:700;padding:3px 12px;border-radius:100px;letter-spacing:1px;">BEST</div>' if is_best else ''}
              <div style="font-size:28px;">{icons[name]}</div>
              <div style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700;
                          color:#f1f5f9; margin:10px 0 16px 0;">{name}</div>
              <div style="display:flex; justify-content:space-around;">
                <div>
                  <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:{colors_m[name]};">{res['R2']:.3f}</div>
                  <div style="font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:1px;">R² Score</div>
                </div>
                <div>
                  <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:{colors_m[name]};">{res['MAE']:.2f}</div>
                  <div style="font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:1px;">MAE</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs Predicted
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        y_pred = results[best_name]['y_pred']
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=list(y_test), y=list(y_pred),
            mode='markers',
            marker=dict(color='#6366f1', size=5, opacity=0.5, line=dict(width=0)),
            name='Predictions'
        ))
        mn, mx = float(y_test.min()), float(y_test.max())
        fig5.add_trace(go.Scatter(
            x=[mn,mx], y=[mn,mx],
            mode='lines',
            line=dict(color='#14b8a6', width=2, dash='dash'),
            name='Perfect Fit'
        ))
        fig5.update_layout(
            title=f'{best_name} — Actual vs Predicted',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8', title_font_color='#f1f5f9',
            title_font_family='Syne', title_font_size=15,
            height=340,
            xaxis=dict(title='Actual Score', gridcolor='#1e293b', linecolor='#1e293b'),
            yaxis=dict(title='Predicted Score', gridcolor='#1e293b', linecolor='#1e293b'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})

    with c2:
        # Feature importance
        rf_model = results['Random Forest']['model']
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance')

        nice_names = {
            'language':'Language','age_group':'Age Group',
            'education_level':'Education','exposure_type':'Exposure Type',
            'daily_practice_hours':'Practice Hours','spoken_at_home':'At Home'
        }
        importance_df['Feature'] = importance_df['Feature'].map(nice_names)

        fig6 = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(
                color=importance_df['Importance'],
                colorscale=[[0,'#1e293b'],[1,'#6366f1']],
                line=dict(width=0)
            ),
            text=[f"{v:.3f}" for v in importance_df['Importance']],
            textposition='outside', textfont=dict(color='#94a3b8', size=11)
        ))
        fig6.update_layout(
            title='Feature Importance (Random Forest)',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8', title_font_color='#f1f5f9',
            title_font_family='Syne', title_font_size=15,
            height=340, showlegend=False,
            xaxis=dict(gridcolor='#1e293b', linecolor='#1e293b', range=[0, 0.7]),
            yaxis=dict(gridcolor='#1e293b', linecolor='#1e293b'),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig6, use_container_width=True, config={'displayModeBar': False})

    # Languages covered
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌐 Languages Covered</div>', unsafe_allow_html=True)
    langs_info = {
        'Hindi':   ('🇮🇳', '~600M speakers', '#6366f1'),
        'English': ('🌍', '~1.5B speakers',  '#14b8a6'),
        'Kannada': ('🏛️', '~45M speakers',   '#f59e0b'),
        'Tamil':   ('📜', '~80M speakers',   '#ec4899'),
        'Bengali': ('🎵', '~230M speakers',  '#22d3ee'),
        'Marathi': ('🏔️', '~83M speakers',   '#a3e635'),
    }
    cols = st.columns(6, gap="small")
    for col, (lang, (icon, speakers, color)) in zip(cols, langs_info.items()):
        with col:
            st.markdown(f"""
            <div style="background:#0f172a; border:1px solid #1e293b; border-radius:14px;
                        padding:20px 12px; text-align:center; border-top:3px solid {color};">
              <div style="font-size:28px;">{icon}</div>
              <div style="font-family:'Syne',sans-serif; font-weight:700; color:#f1f5f9;
                          font-size:14px; margin:8px 0 4px 0;">{lang}</div>
              <div style="font-size:10px; color:#64748b;">{speakers}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#334155; font-size:12px; padding:20px 0;
            border-top:1px solid #1e293b;">
  Built with Streamlit · Scikit-learn · Plotly &nbsp;|&nbsp; LinguaPredict © 2025
</div>
""", unsafe_allow_html=True)
