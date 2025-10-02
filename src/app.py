import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------
# Device
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load pre-trained model for AG News classification
# ----------------------------
@st.cache_resource
def load_model():
    # Using a proper AG News classification model
    try:
        # First try: A model specifically trained for AG News
        model_name = "textattack/distilbert-base-uncased-ag-news"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Verify the model has correct labels
        if hasattr(model.config, 'id2label') and len(model.config.id2label) == 4:
            # Use the model's existing label mapping
            pass
        else:
            # Set up correct AG News label mapping
            model.config.id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
            model.config.label2id = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
            
    except Exception as e1:
        try:
            # Fallback: Another AG News model
            model_name = "fabriceyhc/bert-base-uncased-ag_news"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            # This model uses: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
            model.config.id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
            model.config.label2id = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
        except Exception as e2:
            st.error(f"Failed to load AG News models. Error 1: {e1}, Error 2: {e2}")
            st.error("Please check your internet connection and try again.")
            st.stop()
    
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# Show loading message
with st.spinner("Loading AI model... This may take a moment on first run."):
    tokenizer, model = load_model()
    
st.success("✅ Model loaded successfully!")

# ----------------------------
# Streamlit app
# ----------------------------
st.title("News Category Classification")
st.write("Enter a news headline and description, and the model will predict the category.")

title_input = st.text_input("Title")
desc_input = st.text_area("Description")

if st.button("🔍 Predict Category", type="primary"):
    if title_input.strip() == "" and desc_input.strip() == "":
        st.warning("⚠️ Please enter at least a title or description")
    else:
        with st.spinner("🤖 Analyzing text..."):
            # Combine title and description
            text = f"{title_input.strip()}. {desc_input.strip()}".strip(". ")
            
            # Tokenize with proper settings for news classification
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128  # Full length for better accuracy
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted_class = torch.max(probs, dim=1)

            # Display results
            label_map = {0: "🌍 World", 1: "⚽ Sports", 2: "💼 Business", 3: "🔬 Sci/Tech"}
            category = label_map[int(predicted_class)]
            conf_score = confidence.item()
            
            # Color-coded confidence
            if conf_score > 0.8:
                st.success(f"**Predicted Category:** {category}")
                st.success(f"**Confidence:** {conf_score:.1%} (High)")
            elif conf_score > 0.6:
                st.info(f"**Predicted Category:** {category}")
                st.info(f"**Confidence:** {conf_score:.1%} (Medium)")
            else:
                st.warning(f"**Predicted Category:** {category}")
                st.warning(f"**Confidence:** {conf_score:.1%} (Low)")
                
            # Show all probabilities
            with st.expander("📊 See all category probabilities"):
                for i, (label, prob) in enumerate(zip(label_map.values(), probs[0])):
                    st.write(f"{label}: {prob.item():.1%}")

# Add some helpful examples
st.markdown("---")
st.markdown("### 💡 Try these examples:")
col1, col2 = st.columns(2)

with col1:
    if st.button("📰 World News Example"):
        st.session_state.title_example = "Breaking: International Summit Concludes"
        st.session_state.desc_example = "World leaders reached agreement on climate policies after three days of negotiations."

with col2:
    if st.button("⚽ Sports Example"):
        st.session_state.title_example = "PSG sinks Barcelona with late winner in Champions League"
        st.session_state.desc_example = "PSG's trip to Barcelona was always seen as the standout clash of matchday two. Both teams are expected to go far, with PSG aiming to join Real Madrid as the only other club to successfully defend the trophy in the Champions League era."

# Auto-fill examples if clicked
if hasattr(st.session_state, 'title_example'):
    st.text_input("Title", value=st.session_state.title_example, key="title_auto")
    st.text_area("Description", value=st.session_state.desc_example, key="desc_auto")
