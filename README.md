# 🧠 TweetSense Analyzer

<div align="center">

![TweetSense Banner](https://img.shields.io/badge/NLP-Sentiment_Analysis-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask)

**A Real-Time Twitter Sentiment Analysis Tool powered by Natural Language Processing**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack) • [API Documentation](#-api-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**TweetSense Analyzer** is a comprehensive NLP mini-project that performs sentiment analysis on Twitter data. It classifies text into **Positive**, **Negative**, or **Neutral** sentiments using machine learning models, providing detailed insights through confidence scores, priority classifications, summaries, and visualizations.

### Key Highlights

- 🎭 **Ternary Sentiment Classification** - Analyzes text into three distinct categories
- 📊 **Batch Processing** - Upload CSV files for bulk analysis
- 🎨 **Interactive Visualizations** - Word clouds, pie charts, and bar graphs
- 🔍 **Text Summarization** - Automatic summary generation using LSA
- 🎯 **Priority Classification** - Zero-shot classification for urgency assessment
- 💯 **Confidence Scoring** - Model certainty metrics for each prediction

---

## ✨ Features

### Core Functionality

| Feature                     | Description                                             |
| --------------------------- | ------------------------------------------------------- |
| **Single Text Analysis**    | Analyze individual tweets or text snippets in real-time |
| **Batch CSV Analysis**      | Process multiple records simultaneously from CSV files  |
| **Sentiment Detection**     | Classify text as Positive, Negative, or Neutral         |
| **Confidence Scoring**      | View model prediction probability scores                |
| **Priority Classification** | Categorize text by urgency (High/Medium/Low)            |
| **Text Summarization**      | Generate concise summaries using LSA algorithm          |
| **Word Cloud Generation**   | Visual representation of frequently used terms          |
| **Interactive Dashboard**   | Modern React-based UI with real-time updates            |

### Advanced Analytics

- 📈 **Sentiment Distribution Charts** - Visualize overall sentiment trends
- 📊 **Priority Distribution Graphs** - Analyze urgency patterns
- 📝 **Word Count Analysis** - Track text length statistics
- 🎨 **Base64 Encoded Visualizations** - Embedded image generation
- 💾 **Downloadable Results** - Export analysis as CSV files

---

## 🎬 Demo

### Single Text Analysis

```
Input: "I absolutely love this new feature! It's amazing!"

Output:
├─ Sentiment: Positive
├─ Confidence: 94.3%
├─ Priority: Medium Priority
├─ Word Count: 9 words
└─ Summary: User expresses strong positive sentiment about new feature.
```

### Batch CSV Analysis

```
Input: CSV with 'text' column
Output: Comprehensive report with:
├─ Overall Summary
├─ Sentiment Distribution (Pie Chart)
├─ Priority Distribution (Bar Chart)
├─ Combined Word Cloud
├─ Individual Record Analysis
└─ Downloadable Results CSV
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + TypeScript)           │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐      │
│  │ Input Form │  │ CSV Uploader│  │ Results Dashboard│      │
│  └────────────┘  └─────────────┘  └──────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ (API Calls)
┌─────────────────────────────────────────────────────────────┐
│                    Backend (Flask API)                      │
│  ┌──────────────┐  ┌────────────────┐  ┌───────────────┐    │
│  │ /api/analyze │  │/api/batch_analyze│ │ CORS Support │    │
│  └──────────────┘  └────────────────┘  └───────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    NLP Processing Pipeline                  │
│  ┌──────────────┐  ┌───────────┐  ┌────────────────────┐    │
│  │ Preprocessing│→ │ Prediction│→ │ Post-Processing    │    │
│  │  (spaCy)     │  │ (Logistic)│  │ (TextBlob, Sumy)   │    │
│  └──────────────┘  └───────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm/yarn
- pip package manager

### Backend Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/codaksh7/tweetsense.git
   cd tweetsense-analyzer
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install flask flask-cors joblib spacy pandas textblob sumy transformers wordcloud pillow torch
   ```

4. **Download spaCy model**

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Ensure trained models are present**

   - Place `vectorizer.sav` and `trained_model.sav` in the root directory
   - These should be your pre-trained TF-IDF vectorizer and Logistic Regression model

6. **Run the Flask backend**
   ```bash
   python analysis_backend.py
   ```
   The API will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory**

   ```bash
   cd frontend  # Adjust path based on your structure
   ```

2. **Install dependencies**

   ```bash
   npm install
   # or
   yarn install
   ```

3. **Install required packages**

   ```bash
   npm install lucide-react recharts
   ```

4. **Start development server**
   ```bash
   npm run dev
   # or
   yarn dev
   ```
   The application will be available at `http://localhost:3000`

### Streamlit Alternative (Optional)

If you prefer the Streamlit interface:

```bash
pip install streamlit plotly matplotlib
streamlit run app2.py
```

---

## 📖 Usage

### Single Text Analysis

1. Navigate to the main page
2. Enter your text in the input field
3. Click "Analyze" button
4. View comprehensive results including:
   - Sentiment classification
   - Confidence score
   - Priority level
   - Text summary
   - Word cloud visualization

### Batch CSV Analysis

1. Prepare a CSV file with a column named `text`

   ```csv
   text
   "I love this product!"
   "Terrible experience, very disappointed"
   "It's okay, nothing special"
   ```

2. Click on "Choose CSV File" and select your file
3. Click "Analyze CSV"
4. View comprehensive batch results:
   - Overall summary
   - Sentiment distribution charts
   - Priority distribution
   - Combined word cloud
   - Individual record details
5. Download results as CSV for further analysis

---

## 🛠️ Tech Stack

### Backend

| Technology       | Purpose                                      |
| ---------------- | -------------------------------------------- |
| **Flask**        | REST API framework                           |
| **spaCy**        | Text preprocessing and lemmatization         |
| **scikit-learn** | Machine learning model (Logistic Regression) |
| **Transformers** | Zero-shot classification for priority        |
| **TextBlob**     | Sentiment polarity analysis                  |
| **Sumy**         | Text summarization (LSA algorithm)           |
| **WordCloud**    | Visual word frequency generation             |
| **Pandas**       | Data manipulation for batch processing       |

### Frontend

| Technology       | Purpose                         |
| ---------------- | ------------------------------- |
| **React 18**     | UI framework                    |
| **TypeScript**   | Type-safe development           |
| **Tailwind CSS** | Styling and responsive design   |
| **Lucide React** | Icon library                    |
| **Recharts**     | Data visualization (charts)     |
| **Next.js**      | React framework (if applicable) |

### Alternative UI

| Technology     | Purpose                     |
| -------------- | --------------------------- |
| **Streamlit**  | Rapid prototyping dashboard |
| **Plotly**     | Interactive visualizations  |
| **Matplotlib** | Statistical plots           |

---

## 📡 API Documentation

### Base URL

```
http://localhost:5000/api
```

### Endpoints

#### 1. Single Text Analysis

**Endpoint:** `POST /analyze`

**Request Body:**

```json
{
  "text": "I absolutely love this new feature!"
}
```

**Response:**

```json
{
  "status": "success",
  "sentiment": "Positive",
  "confidence": "0.9431",
  "priority": "Medium Priority",
  "input_length": 7,
  "summary": "User expresses positive sentiment about feature.",
  "wordcloud_img": "data:image/png;base64,iVBORw0KG..."
}
```

#### 2. Batch CSV Analysis

**Endpoint:** `POST /batch_analyze`

**Request:** Multipart form data with file upload

**Response:**

```json
{
  "status": "success",
  "summary": {
    "total_records": 100,
    "overall_summary": "Mixed sentiments with majority positive...",
    "sentiment_distribution": {
      "Positive": 65,
      "Negative": 20,
      "Neutral": 15
    },
    "priority_distribution": {
      "High Priority": 30,
      "Medium Priority": 50,
      "Low Priority": 20
    },
    "avg_confidence": 0.8723,
    "avg_word_count": 15.4,
    "wordcloud_img": "data:image/png;base64,..."
  },
  "individual_results": [
    {
      "text": "Sample tweet text",
      "sentiment": "Positive",
      "confidence": 0.89,
      "priority": "Medium Priority",
      "word_count": 12
    }
  ]
}
```

### Error Responses

```json
{
  "status": "error",
  "message": "Error description here"
}
```

**Status Codes:**

- `200` - Success
- `400` - Bad Request (missing/invalid parameters)
- `500` - Internal Server Error

---

## 📁 Project Structure

```
tweetsense/
│
├── backend/
│   ├── analysis_backend.py      # Flask REST API backend
│   ├── app2.py                   # Streamlit alternative UI
│   ├── proj.ipynb                # Jupyter notebook (model training/exploration)
│   ├── trained_model.sav         # Trained Logistic Regression model
│   └── vectorizer.sav            # Trained TF-IDF vectorizer
│
├── frontend/
│   ├── app/
│   │   ├── favicon.ico           # App favicon
│   │   ├── globals.css           # Global styles
│   │   ├── layout.tsx            # Root layout component
│   │   └── page.tsx              # Main page component
│   │
│   ├── components/
│   │   └── SentimentHomePage.tsx # Main sentiment analysis component
│   │
│   ├── node_modules/             # npm dependencies
│   ├── public/                   # Static assets
│   │
│   ├── .gitignore                # Git ignore rules
│   ├── eslint.config.mjs         # ESLint configuration
│   ├── next-env.d.ts             # Next.js TypeScript declarations
│   ├── next.config.ts            # Next.js configuration
│   ├── package-lock.json         # npm lock file
│   ├── package.json              # npm dependencies & scripts
│   ├── postcss.config.mjs        # PostCSS configuration
│   ├── README.md                 # Frontend documentation
│   └── tsconfig.json             # TypeScript configuration
│
└── README.md                     # Main project documentation
```

### Key Files Description

| File/Folder                                 | Description                                                           |
| ------------------------------------------- | --------------------------------------------------------------------- |
| `backend/analysis_backend.py`               | Flask REST API with `/api/analyze` and `/api/batch_analyze` endpoints |
| `backend/app2.py`                           | Streamlit-based alternative UI for quick prototyping                  |
| `backend/proj.ipynb`                        | Jupyter notebook for model training and data exploration              |
| `backend/trained_model.sav`                 | Serialized Logistic Regression model                                  |
| `backend/vectorizer.sav`                    | Serialized TF-IDF vectorizer                                          |
| `frontend/components/SentimentHomePage.tsx` | Main React component with UI and API integration                      |
| `frontend/app/page.tsx`                     | Next.js page entry point                                              |
| `frontend/next.config.ts`                   | Next.js framework configuration                                       |
| `frontend/package.json`                     | Frontend dependencies (React, Recharts, Lucide, etc.)                 |

## 🧪 Model Details

### Sentiment Classification Model

- **Algorithm:** Logistic Regression
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Preprocessing:**
  - Tokenization using spaCy
  - Lemmatization
  - Stopword removal (implicit in model training)
- **Classes:** 3 (Positive, Negative, Neutral)
- **Output:** Sentiment label + confidence score

### Priority Classification

- **Model:** Zero-Shot Classification (Transformers)
- **Labels:** High Priority, Medium Priority, Low Priority
- **Purpose:** Assess urgency/importance of text content

### Text Summarization

- **Algorithm:** LSA (Latent Semantic Analysis)
- **Library:** Sumy
- **Output:** 2-3 sentence summary of input text

### Sentiment Polarity Enhancement

- **Tool:** TextBlob
- **Purpose:** Refine sentiment classification using polarity scores
- **Threshold:** -0.1 to 0.1 for Neutral classification

## 🎨 Features Showcase

### Visualizations

1. **Pie Charts** - Model confidence breakdown
2. **Bar Charts** - Priority distribution analysis
3. **Word Clouds** - Visual frequency representation
4. **Table Views** - Detailed record-by-record results

### User Experience

- ✅ Real-time analysis feedback
- ✅ Loading states and animations
- ✅ Error handling with user-friendly messages
- ✅ Responsive design for all devices
- ✅ Dark mode support (Streamlit version)
- ✅ Downloadable results

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- [ ] Add more ML models (BERT, RoBERTa)
- [ ] Implement caching for faster responses
- [ ] Add authentication and user management
- [ ] Create Docker containerization
- [ ] Add more visualization options
- [ ] Implement real-time Twitter API integration
- [ ] Add multi-language support

## 👥 Team/Authors

**Daksh Thakkar**

- GitHub: [@codaksh7](https://github.com/codaksh7)
- Email: dakshthakkar296@gmail.com

**Aryan Verma**

- GitHub: [@aryan-oo9](https://github.com/aryan-oo9)
- Email: aryanverma1750@gmail.com

## 🙏 Acknowledgments

- Dataset: [Twitter Sentiment Analysis Dataset]
- spaCy for NLP preprocessing
- Hugging Face Transformers for zero-shot classification
- The open-source community

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/codaksh7/tweetsense/issues) page
2. Create a new issue with detailed description
3. Contact the team via email

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made in 2025

</div>
