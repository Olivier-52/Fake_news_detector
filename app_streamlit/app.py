import streamlit as st
import requests

if __name__ == "__main__":

    st.set_page_config(
        page_title="Detection des fausses nouvelles sur le changement climatique",
        page_icon="🌍",
        layout="wide"
    )

    st.header("Detection des fausses nouvelles sur le changement climatique")
    st.markdown("👋 Bienvenue dans sur le detecteur de fausses nouvelles sur le réchauffement climatique ! Cette application vous permet de vérifier la véracité des informations relatives au changement climatique.")
    st.caption("Cette application utilise un modèle d'apprentissage automatique pour classer les articles d'actualité comme vrais ou faux en ce qui concerne le changement climatique.")
    user_input = st.text_area("Veuillez entrer le texte de l'article:", height=200)

    if st.button("Vérifier la nouvelle"):
        if user_input.strip() == "":
            st.warning("Veuillez saisir un texte avant de vérifier.")
        else:
            with st.spinner("Analyse de l'article en cours..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/predict",
                        json={"text": user_input}
                    )
                    response.raise_for_status()
                    result = response.json()
                    prediction = result.get("prediction", "unknown")

                    if prediction == "real":
                        st.success("L'article est probablement vrai.")
                    elif prediction == "fake":
                        st.error("L'article est probablement faux.")
                    elif prediction == "biased":
                        st.warning("L'article est probablement Vrai, mais biaisé.")
                    else:
                        st.info("L'article n'a pas pu être classé.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Une erreur est survenue lors de la vérification de l'article: {e}")
        