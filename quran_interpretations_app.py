import streamlit as st
import pandas as pd
import requests
import io

REPO_BASE = "https://raw.githubusercontent.com/jsnhum/LLM.-Qur-an-translation/main"

CSV_FILES = [
    "Claude_haiku.csv",
    "Claude_opus.csv",
    "Claude_sonnet.csv",
    "Claude_sonnet_3.5.csv",
    "GPT3.csv",
    "GPT4Turbo.csv",
    "GPT4o.csv",
    "GPT4oMini.csv",
    "Grok2.csv",
    "Llama2.csv",
    "Llama3.csv",
    "Mixtral.csv",
    "gemini_1.5_flash.csv",
    "gemini_1.5_pro.csv",
    "gemini_2.0_flash.csv",
]

MODEL_DISPLAY_NAMES = {
    "Claude_haiku": "Claude Haiku",
    "Claude_opus": "Claude Opus",
    "Claude_sonnet": "Claude Sonnet",
    "Claude_sonnet_3.5": "Claude Sonnet 3.5",
    "GPT3": "GPT-3",
    "GPT4Turbo": "GPT-4 Turbo",
    "GPT4o": "GPT-4o",
    "GPT4oMini": "GPT-4o Mini",
    "Grok2": "Grok 2",
    "Llama2": "Llama 2",
    "Llama3": "Llama 3",
    "Mixtral": "Mixtral",
    "gemini_1.5_flash": "Gemini 1.5 Flash",
    "gemini_1.5_pro": "Gemini 1.5 Pro",
    "gemini_2.0_flash": "Gemini 2.0 Flash",
}


@st.cache_data(show_spinner="Loading Qur'an translations from GitHub...")
def load_all_data():
    """Fetch all CSV files from GitHub and combine into a single DataFrame."""
    frames = []
    errors = []

    for filename in CSV_FILES:
        url = f"{REPO_BASE}/{filename}"
        model_key = filename.removesuffix(".csv")
        display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), index_col=0)
            df["Model"] = display_name
            frames.append(df)
        except Exception as e:
            errors.append(f"{display_name}: {e}")

    if not frames:
        st.error("Failed to load any data. Check your internet connection.")
        st.stop()

    combined = pd.concat(frames, ignore_index=True)

    # Normalise column names to lowercase
    combined.columns = [c.strip().lower() for c in combined.columns]

    # Ensure chapter and verse are integers
    combined["chapter"] = pd.to_numeric(combined["chapter"], errors="coerce")
    combined["verse"] = pd.to_numeric(combined["verse"], errors="coerce")
    combined.dropna(subset=["chapter", "verse"], inplace=True)
    combined["chapter"] = combined["chapter"].astype(int)
    combined["verse"] = combined["verse"].astype(int)

    return combined, errors


def main():
    st.set_page_config(
        page_title="LLM Qur'an Translations",
        page_icon="📖",
        layout="wide",
    )

    st.title("QAHA-Qur'anic Artificial Hermeneutics Archive")
    st.markdown(
        "Compare how different Large Language Models translate and interpret "
        "verses of the Qur'an.  \n"
        "Data source: "
        "[jsnhum/LLM.-Qur-an-translation]"
        "(https://github.com/jsnhum/LLM.-Qur-an-translation)"
    )

    # --- Load data ---
    data, load_errors = load_all_data()

    if load_errors:
        with st.expander(f"{len(load_errors)} model(s) failed to load"):
            for err in load_errors:
                st.warning(err)

    # --- Sidebar selectors ---
    st.sidebar.header("Select verse")

    chapters = sorted(data["chapter"].unique())
    selected_chapter = st.sidebar.selectbox(
        "Chapter (Surah)",
        chapters,
        format_func=lambda x: f"Surah {x}",
    )

    verses_in_chapter = sorted(
        data.loc[data["chapter"] == selected_chapter, "verse"].unique()
    )
    selected_verse = st.sidebar.selectbox(
        "Verse (Ayah)",
        verses_in_chapter,
        format_func=lambda x: f"Ayah {x}",
    )

    # --- Model selector ---
    st.sidebar.header("Select models")

    all_models = sorted(data["model"].unique())
    select_all = st.sidebar.checkbox("Select all models", value=True)

    if select_all:
        selected_models = st.sidebar.multiselect(
            "Models",
            all_models,
            default=all_models,
        )
    else:
        selected_models = st.sidebar.multiselect(
            "Models",
            all_models,
        )

    if not selected_models:
        st.info("Please select at least one model from the sidebar.")
        return

    # --- Filter data ---
    mask = (
        (data["chapter"] == selected_chapter)
        & (data["verse"] == selected_verse)
        & (data["model"].isin(selected_models))
    )
    verse_data = data.loc[mask]

    # --- Display Arabic original (same for all models) ---
    if "orig" in verse_data.columns and not verse_data["orig"].dropna().empty:
        arabic_text = verse_data["orig"].dropna().iloc[0].strip()
        st.markdown(
            f'<p style="text-align:right; font-size:1.6rem; '
            f'font-family:serif; line-height:2.2;" dir="rtl">'
            f"{arabic_text}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

    st.subheader(f"Surah {selected_chapter}, Ayah {selected_verse}")

    # --- Render each model's output ---
    for _, row in verse_data.sort_values("model").iterrows():
        model_name = row["model"]

        st.markdown(f"### {model_name}")

        translation = row.get("translation", "")
        interpretation = row.get("interpretation", "")

        if pd.notna(translation) and str(translation).strip():
            st.markdown("**Translation**")
            st.markdown(f"> {str(translation).strip()}")

        if pd.notna(interpretation) and str(interpretation).strip():
            st.markdown("**Interpretation**")
            st.markdown(str(interpretation).strip())

        st.markdown("---")


if __name__ == "__main__":
    main()
