
import os
import joblib
import streamlit as st

from hallucination_checker import EvidenceChecker
from evidence_retriever import retrieve_evidence


PREFERENCE_MODEL_PATH = "models/preference_model.joblib"

st.set_page_config(
    page_title="LLM Response Quality Checker",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LLM Response Quality Checker")

st.write(
    "Compare AI responses and screen factual claims "
    "against reference evidence."
)


@st.cache_resource
def load_evidence_checker():
    return EvidenceChecker()


@st.cache_resource
def load_preference_model():
    return joblib.load(PREFERENCE_MODEL_PATH)


tab1, tab2, tab3 = st.tabs([
    "Compare responses",
    "Check provided evidence",
    "Automatic evidence retrieval",
])


# --------------------------------------------------
# TAB 1: Compare two responses
# --------------------------------------------------

with tab1:
    st.subheader("Human-preference prediction")

    if not os.path.exists(PREFERENCE_MODEL_PATH):
        st.info(
            "Train your preference model first using "
            "`python train.py`."
        )
    else:
        model = load_preference_model()

        prompt = st.text_area(
            "Prompt",
            key="preference_prompt",
        )

        col_a, col_b = st.columns(2)

        with col_a:
            response_a = st.text_area(
                "Response A",
                height=180,
                key="response_a",
            )

        with col_b:
            response_b = st.text_area(
                "Response B",
                height=180,
                key="response_b",
            )

        if st.button("Compare responses"):
            if not all([
                prompt.strip(),
                response_a.strip(),
                response_b.strip(),
            ]):
                st.warning(
                    "Enter the prompt and both responses."
                )
            else:
                combined = (
                    f"PROMPT: {prompt.strip()} "
                    f"RESPONSE A: {response_a.strip()} "
                    f"RESPONSE B: {response_b.strip()}"
                )

                prediction = model.predict([combined])[0]

                st.success(
                    f"Predicted preference: {prediction}"
                )

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(
                        [combined]
                    )[0]

                    labels = model.named_steps[
                        "classifier"
                    ].classes_

                    st.caption(
                        "These scores measure preference-model "
                        "output, not factual accuracy."
                    )

                    for label, score in sorted(
                        zip(labels, probabilities),
                        key=lambda item: item[1],
                        reverse=True,
                    ):
                        st.write(f"{label}: {score:.1%}")


# --------------------------------------------------
# TAB 2: Check evidence supplied by the user
# --------------------------------------------------

with tab2:
    st.subheader("Check against provided evidence")

    evidence = st.text_area(
        "Trusted reference evidence",
        height=160,
        key="manual_evidence",
    )

    answer = st.text_area(
        "AI response",
        height=160,
        key="manual_answer",
    )

    if st.button("Check provided evidence"):
        if not evidence.strip() or not answer.strip():
            st.warning(
                "Provide both reference evidence and an AI response."
            )
        else:
            try:
                with st.spinner("Checking claims..."):
                    checker = load_evidence_checker()
                    results = checker.check_answer(
                        evidence.strip(),
                        answer.strip(),
                    )

                for index, result in enumerate(results, 1):
                    st.markdown(
                        f"**Claim {index}:** {result['claim']}"
                    )
                    st.write(
                        f"**Result:** {result['status']}"
                    )
                    st.write(
                        "Model score: "
                        f"{result['confidence']:.1%}"
                    )

                    st.divider()

            except Exception as error:
                st.error(f"Could not check claims: {error}")


# --------------------------------------------------
# TAB 3: Automatically retrieve evidence
# --------------------------------------------------

with tab3:
    st.subheader("Automatic evidence retrieval")

    st.write(
        "Search Wikipedia for reference material, "
        "then compare each response sentence with the retrieved text."
    )

    topic = st.text_input(
        "Optional topic to improve search",
        placeholder="Example: Australia",
    )

    auto_answer = st.text_area(
        "AI response to verify",
        height=180,
        key="automatic_answer",
    )

    if st.button("Retrieve sources and check"):
        if not auto_answer.strip():
            st.warning("Enter an AI response first.")
        else:
            try:
                checker = load_evidence_checker()
                claims = checker.split_claims(
                    auto_answer.strip()
                )

                for index, claim in enumerate(claims, 1):
                    query = (
                        f"{topic.strip()} {claim}".strip()
                        if topic.strip()
                        else claim
                    )

                    st.markdown(f"### Claim {index}")
                    st.write(claim)

                    with st.spinner("Searching Wikipedia..."):
                        sources = retrieve_evidence(
                            query,
                            max_results=3,
                        )

                    if not sources:
                        st.warning(
                            "No evidence found. "
                            "This does not mean the claim is false."
                        )
                        continue

                    evidence_text = "\n\n".join(
                        f"Source: {source['title']}\n"
                        f"{source['extract']}"
                        for source in sources
                    )

                    with st.spinner("Checking evidence..."):
                        result = checker.check_claim(
                            evidence_text,
                            claim,
                        )

                    st.write(
                        f"**Screening result:** {result['status']}"
                    )

                    st.write(
                        "Model score: "
                        f"{result['confidence']:.1%}"
                    )

                    st.markdown("**Sources retrieved**")

                    for source in sources:
                        st.markdown(
                            f"- [{source['title']}]"
                            f"({source['url']})"
                        )

                        st.caption(
                            source["extract"][:700]
                            + (
                                "..."
                                if len(source["extract"]) > 700
                                else ""
                            )
                        )

                    st.divider()

                st.warning(
                    "Experimental screening only. Retrieved pages "
                    "may be irrelevant or incomplete. Check the linked "
                    "sources yourself, especially for important claims."
                )

            except Exception as error:
                st.error(
                    f"Automatic evidence retrieval failed: {error}"
                )
                st.info(
                    "Check your internet connection and installed packages."
                )
