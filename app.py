import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Streamlit app title
st.title("Distillation Binaire : Méthode de McCabe - Thiele")

# Input parameters
xF = st.number_input("Fraction molaire du composant léger dans l'alimentation (xF)", value=0.10, min_value=0.0, max_value=1.0)
xD = st.number_input("Fraction molaire du composant léger dans le distillat (xD)", value=0.80, min_value=0.0, max_value=1.0)
xW = st.number_input("Fraction molaire du composant léger dans le rebouilleur (xW)", value=0.10, min_value=0.0, max_value=1.0)
R_factor = st.number_input("Facteur du ratio de reflux (L/D)", value=1.5, min_value=0.0)
a = st.number_input("Facteur de séparation (a)", value=2.5, min_value=0.0)
q = st.number_input("Condition d'alimentation (q)", value=1.5, min_value=0.0)

# Check for valid q value
if q <= 0 or q == 1:
    st.error("La condition d'alimentation (q) doit être supérieure à 0 et différente de 1.")
else:
    # Courbe d'équilibre
    def eq_curve(a):
        x_eq = np.linspace(0, 1, 51)
        y_eq = a * x_eq / (1 + (a - 1) * x_eq)
        return y_eq, x_eq

    y_eq, x_eq = eq_curve(a)

    # Ligne d'alimentation
    def fed(xF, q, a):    
        c1 = (q * (a - 1))
        c2 = q + xF * (1 - a) - a * (q - 1)
        c3 = -xF
        coeff = [c1, c2, c3]
        r = np.sort(np.roots(coeff))
        
        xiE = r[0] if r[0] > 0 else r[1]
        yiE = a * xiE / (1 + xiE * (a - 1))
        
        if q == 1:
            x_fed = [xF, xF]
            y_fed = [xF, yiE]
        else:
            x_fed = np.linspace(xF, xiE, 51)
            y_fed = q / (q - 1) * x_fed - xF / (q - 1)
        
        return xiE, yiE, y_fed, x_fed

    xiE, yiE, y_fed, x_fed = fed(xF, q, a)

    # Calcul de R_min et R
    R_min = (xD - yiE) / (yiE - xiE)
    R = R_factor * R_min

    # Point d'alimentation
    xiF = (xF / (q - 1) + xD / (R + 1)) / (q / (q - 1) - R / (R + 1))
    yiF = R / (R + 1) * xiF + xD / (R + 1)

    # Section de rectification
    def rect(R, xD, xiF):
        x_rect = np.linspace(xiF - 0.025, xD, 51)    
        y_rect = R / (R + 1) * x_rect + xD / (R + 1)
        return y_rect, x_rect

    y_rect, x_rect = rect(R, xD, xiF)

    # Section de stripping
    def stp(xiF, yiF, xW):
        x_stp = np.linspace(xW, xiF + 0.025, 51)    
        y_stp = ((yiF - xW) / (xiF - xW)) * (x_stp - xW) + xW
        return y_stp, x_stp

    y_stp, x_stp = stp(xiF, yiF, xW)

    # Construction des étages
    s = np.zeros((1000, 5))
    for i in range(1, 1000):
        s[0, 0] = xD
        s[0, 1] = xD
        s[0, 2] = s[0, 1] / (a - s[0, 1] * (a - 1))
        s[0, 3] = s[0, 1]
        s[0, 4] = 0
        
        s[i, 0] = s[i - 1, 2]
        
        if s[i, 0] < xW:
            s[i, 1] = s[i, 0] 
            s[i, 2] = s[i, 0]
            s[i, 3] = s[i, 0]
            s[i, 4] = i
            break  
        if s[i, 0] > xiF:
            s[i, 1] = R / (R + 1) * s[i, 0] + xD / (R + 1)
        elif s[i, 0] < xiF:
            s[i, 1] = ((yiF - xW) / (xiF - xW)) * (s[i, 0] - xW) + xW
        else:
            s[i, 1] = s[i - 1, 3]
        
        if s[i, 0] > xW:
            s[i, 2] = s[i, 1] / (a - s[i, 1] * (a - 1))
        else:
            s[i, 2] = s[i, 0]
        
        s[i, 3] = s[i, 1]
        
        if s[i, 0] < xiF:
            s[i, 4] = i
        else:
            s[i, 4] = 0

    s = s[~np.all(s == 0, axis=1)]
    s_rows = np.size(s, 0)
    S = np.zeros((s_rows * 2, 2))

    for i in range(0, s_rows):
        S[i * 2, 0] = s[i, 0]
        S[i * 2, 1] = s[i, 1]
        S[i * 2 + 1, 0] = s[i, 2]
        S[i * 2 + 1, 1] = s[i, 3]

    # Numérotation des étages
    x_s = s[:, 2:3]
    y_s = s[:, 3:4]
    stage = np.char.mod('%d', np.linspace(1, s_rows - 1, s_rows - 1))

    # Emplacement de la plaque d'alimentation
    s_f = s_rows - np.count_nonzero(s[:, 4:5], axis=0)
    s_f_scalar = s_f.item()

    # Tracé avec Plotly
    fig = go.Figure()

    # Ajouter les courbes
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Ligne d'équilibre", line=dict(color='black')))
    fig.add_trace(go.Scatter(x=x_eq, y=y_eq, mode='lines', name="Courbe d'équilibre", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_rect, y=y_rect, mode='lines', name="Section de rectification OL", line=dict(dash='dash', color='black')))
    fig.add_trace(go.Scatter(x=x_stp, y=y_stp, mode='lines', name="Section de stripping OL", line=dict(dash='dot', color='black')))
    fig.add_trace(go.Scatter(x=x_fed, y=y_fed, mode='lines', name="Ligne d'alimentation", line=dict(dash='dot', color='black')))
    fig.add_trace(go.Scatter(x=S[:, 0], y=S[:, 1], mode='lines', name="Étages", line=dict(color='blue')))

    # Numéros d'étages
    for label, x, y in zip(stage, x_s.flatten(), y_s.flatten()):
        fig.add_annotation(x=x, y=y + 0.02, text=label, showarrow=False, font=dict(size=10))

    # Points d'alimentation, de distillat et de rebouilleur
    fig.add_trace(go.Scatter(x=[xF, xD, xW], y=[xF, xD, xW], mode='markers', marker=dict(color='green', size=10), name='Points'))
    fig.add_annotation(x=xF + 0.05, y=xF - 0.03, text='($x_{F}, x_{F}$)', showarrow=False)
    fig.add_annotation(x=xD + 0.05, y=xD - 0.03, text='($x_{D}, x_{D}$)', showarrow=False)
    fig.add_annotation(x=xW + 0.05, y=xW - 0.03, text='($x_{W}, x_{W}$)', showarrow=False)

    # Mises à jour de l'axe
    fig.update_layout(
        title="Méthode de McCabe-Thiele pour la distillation binaire",
        xaxis_title="Fraction molaire du composant léger",
        yaxis_title="Fraction molaire du composant léger",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.7, y=0.1)
    )

    # Affichage des résultats
    st.plotly_chart(fig)
    st.write(f"**Rmin**: {R_min:.2f}")
    st.write(f"**R**: {R:.2f}")
    st.write(f"**Nombre d'étages**: {s_rows - 1}")
    st.write(f"**Étages de l'alimentation**: {s_f_scalar}")
