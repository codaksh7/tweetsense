"use client";

import React, { useState } from "react";
import {
  Search,
  ArrowRight,
  Zap,
  MessageSquare,
  Twitter,
  Code,
  Lightbulb,
  ClipboardCheck,
  BarChart3,
  Menu,
  X,
  Brain,
  TrendingUp,
  TrendingDown,
  Target,
  FlaskConical,
  MessageCircle, // Added for summary icon
  Cloud, // Added for word cloud icon
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";

// --- API Configuration ---
const API_URL = "http://localhost:5000/api/analyze";

// --- UPDATED TYPES ---
interface AnalysisResult {
  sentiment: "Positive" | "Negative" | "Neutral";
  confidence: string; // e.g., "0.9543"
  priority: string; // e.g., "High Priority"
  input_length: number;
  summary: string; // NEW FIELD for text summary
  wordcloud_img: string; // NEW FIELD for Base64 image string
}

const getSentimentColor = (sentiment: string) => {
  switch (sentiment) {
    case "Positive":
      return "#10B981";
    case "Negative":
      return "#EF4444";
    case "Neutral":
      return "#3B82F6";
    default:
      return "#6B7280";
  }
};

const SentimentHomePage: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);

  const handleAnalysis = async (): Promise<void> => {
    if (searchQuery.trim().length < 5) {
      setError(
        "Please enter a valid phrase or hashtag of sufficient length for analysis."
      );
      setResult(null);
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: searchQuery }),
      });

      const data = await response.json();

      if (response.ok && data.status === "success") {
        // Cast the data to the new interface
        setResult(data as AnalysisResult);

        setTimeout(() => {
          document
            .getElementById("results")
            ?.scrollIntoView({ behavior: "smooth" });
        }, 100);
      } else {
        setError(
          data.message || "Analysis failed due to an unknown model error."
        );
      }
    } catch (err) {
      console.error("Fetch Error:", err);
      setError(
        `Failed to connect to the backend API at ${API_URL}. Please ensure your Python Flask server is running.`
      );
    } finally {
      setLoading(false);
    }
  };

  const confidenceValue = result ? parseFloat(result.confidence) * 100 : 0;
  const chartData = result
    ? [
        {
          name: `${result.sentiment} (${confidenceValue.toFixed(1)}%)`,
          value: confidenceValue,
          color: getSentimentColor(result.sentiment),
        },
        {
          name: `Uncertainty (${(100 - confidenceValue).toFixed(1)}%)`,
          value: 100 - confidenceValue,
          color: "#D1D5DB",
        },
      ]
    : [];

  const AnalysisStatCard = ({
    title,
    value,
    icon,
    color,
  }: {
    title: string;
    value: string | number;
    icon: React.ReactNode;
    color: string;
  }) => (
    <div
      className={`bg-white p-6 rounded-xl shadow-lg border border-gray-100 flex items-center space-x-4 hover:shadow-xl transition-shadow`}
    >
      <div
        className={`p-3 rounded-full`}
        style={{ backgroundColor: `${color}1A`, color: color }}
      >
        {icon}
      </div>
      <div>
        <div className="text-sm font-medium text-gray-600">{title}</div>
        <div className="text-3xl font-bold" style={{ color: color }}>
          {value}
        </div>
      </div>
    </div>
  );

  const LoadingSpinner = () => (
    <div className="flex justify-center items-center py-6">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-600"></div>
      <p className="text-gray-600 ml-3">
        Processing with Logistic Regression...
      </p>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-green-50">
      {/* Header */}
      <header className="bg-white shadow-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-orange-500 to-green-600 p-2 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-orange-600 to-green-700 bg-clip-text text-transparent">
                  TweetSense Analyzer
                </h1>
                <p className="text-xs text-gray-500 hidden sm:block">
                  NLP Mini Project: Real-Time Sentiment
                </p>
              </div>
            </div>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center space-x-8">
              <a
                href="#features"
                className="text-gray-700 hover:text-orange-600 font-medium transition-colors"
              >
                Features
              </a>
              <a
                href="#results"
                className="text-gray-700 hover:text-orange-600 font-medium transition-colors"
              >
                Demo
              </a>
              <a
                href="https://github.com/your-repo-link"
                target="_blank"
                rel="noopener noreferrer"
                className="bg-gradient-to-r from-orange-500 to-red-600 text-white px-6 py-2 rounded-lg font-semibold hover:shadow-lg transition-all duration-300 transform hover:-translate-y-0.5"
              >
                GitHub Repo
              </a>
            </nav>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100"
            >
              {isMobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>

          {/* Mobile Menu Content */}
          {isMobileMenuOpen && (
            <div className="md:hidden py-4 border-t border-gray-200">
              <div className="flex flex-col space-y-4">
                <a
                  href="#features"
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="text-gray-700 hover:text-orange-600 font-medium transition-colors"
                >
                  Features
                </a>
                <a
                  href="#results"
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="text-gray-700 hover:text-orange-600 font-medium transition-colors"
                >
                  Demo
                </a>
                <a
                  href="https://github.com/your-repo-link"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="bg-gradient-to-r from-orange-500 to-red-600 text-white px-6 py-2 rounded-lg font-semibold hover:shadow-lg transition-all duration-300 transform hover:-translate-y-0.5 self-start"
                >
                  GitHub Repo
                </a>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-orange-100 to-green-100 rounded-full text-orange-800 text-sm font-medium mb-6">
            <Twitter className="w-4 h-4 mr-2" />
            Analyze the Digital Pulse of the Crowd
          </div>

          <h1 className="text-5xl md:text-7xl font-bold text-gray-900 mb-6">
            <span className="bg-gradient-to-r from-orange-600 via-red-500 to-green-600 bg-clip-text text-transparent">
              TweetSense:
            </span>
            <br />
            Real-Time Twitter Sentiment
          </h1>

          <p className="text-xl text-gray-600 mb-10 max-w-4xl mx-auto leading-relaxed">
            Harness the power of Natural Language Processing (NLP) to classify
            the mood and opinion hidden within Twitter conversations. Paste a
            phrase or hashtag below to analyze instantly.
          </p>

          {/* Analysis Input */}
          <div className="max-w-3xl mx-auto mb-8">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Paste Tweet or Topic (e.g., I love this new feature)"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-12 pr-4 py-4 text-lg border-2 border-gray-200 rounded-xl focus:border-orange-500 focus:outline-none shadow-lg"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleAnalysis();
                }}
                disabled={loading}
              />
              <button
                onClick={handleAnalysis}
                disabled={loading}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-gradient-to-r from-orange-500 to-red-600 text-white px-6 py-2 rounded-lg font-semibold hover:shadow-lg transition-all disabled:opacity-50"
              >
                {loading ? "Analyzing..." : "Analyze"}
              </button>
            </div>

            {/* Error/Loading Message */}
            {loading && <LoadingSpinner />}
            {error && (
              <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded-lg text-sm text-left">
                **Error:** {error}
              </div>
            )}
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={handleAnalysis}
              disabled={loading}
              className="bg-gradient-to-r from-green-600 to-blue-600 text-white px-8 py-4 rounded-xl font-semibold hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 flex items-center justify-center disabled:opacity-50"
            >
              Start Sentiment Analysis
              <Zap className="ml-2 w-5 h-5" />
            </button>
            <a
              href="https://github.com/your-repo-link"
              target="_blank"
              rel="noopener noreferrer"
              className="border-2 border-gray-300 text-gray-700 px-8 py-4 rounded-xl font-semibold hover:border-orange-500 hover:text-orange-600 transition-all duration-300 flex items-center justify-center"
            >
              <Code className="mr-2 w-5 h-5" />
              View Source Code
            </a>
          </div>
        </div>
      </section>

      {/* Analysis Results Section (DYNAMIC) */}
      {result && (
        <section id="results" className="py-20 bg-blue-50/50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-gray-900 mb-4">
                Analysis Results
              </h2>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                Insights generated by **Logistic Regression** and **Zero-Shot
                Classification** models.
              </p>
            </div>

            <div className="grid lg:grid-cols-3 gap-8">
              {/* Result Card: Sentiment */}
              <AnalysisStatCard
                title="Predicted Sentiment"
                value={result.sentiment}
                icon={
                  result.sentiment === "Positive" ? (
                    <TrendingUp className="w-5 h-5" />
                  ) : result.sentiment === "Negative" ? (
                    <TrendingDown className="w-5 h-5" />
                  ) : (
                    <MessageSquare className="w-5 h-5" />
                  )
                }
                color={getSentimentColor(result.sentiment)}
              />
              {/* Result Card: Confidence */}
              <AnalysisStatCard
                title="Model Confidence"
                value={`${(parseFloat(result.confidence) * 100).toFixed(2)}%`}
                icon={<FlaskConical className="w-5 h-5" />}
                color="#6366F1"
              />
              {/* Result Card: Priority */}
              <AnalysisStatCard
                title="Priority Classification"
                value={result.priority}
                icon={<Target className="w-5 h-5" />}
                color="#F59E0B"
              />
            </div>

            <div className="grid lg:grid-cols-2 gap-8 mt-8">
              {/* Pie Chart Visualization (Confidence Breakdown) */}
              <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
                <h3 className="text-xl font-bold text-gray-900 mb-4">
                  Model Certainty
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={chartData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label={({ name }) => name}
                    >
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(value: number) => [
                        `${value.toFixed(2)}%`,
                        "Share",
                      ]}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Input Details */}
              <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
                <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                  <ClipboardCheck className="w-5 h-5 mr-2 text-gray-600" />
                  Input Details
                </h3>
                <div className="space-y-4">
                  <p className="text-gray-600 italic">
                    The prediction was based on the provided text, which was
                    tokenized and lemmatized before being passed to the model.
                  </p>
                  <div className="bg-gray-50 p-3 rounded-lg text-sm">
                    <p className="text-gray-800 font-semibold mb-1">
                      Raw Input:
                    </p>
                    <p className="text-gray-700 break-words">{searchQuery}</p>
                  </div>
                  <div className="flex justify-between items-center bg-gray-50 p-3 rounded-lg text-sm">
                    <span className="font-semibold text-gray-800">
                      Word Count:
                    </span>
                    <span className="text-gray-600">{result.input_length}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* NEW SECTION: Summary and Word Cloud */}
            <div className="grid md:grid-cols-2 gap-8 mt-8">
              {/* Text Summary */}
              <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
                <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                  <MessageCircle className="w-5 h-5 mr-2 text-gray-600" />
                  Generated Summary
                </h3>
                <p className="text-gray-700 leading-relaxed italic">
                  {result.summary}
                </p>
              </div>

              {/* Word Cloud */}
              <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-100 flex flex-col items-center">
                <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                  <Cloud className="w-5 h-5 mr-2 text-gray-600" />
                  Word Cloud
                </h3>
                {result.wordcloud_img ? (
                  <img
                    src={result.wordcloud_img}
                    alt="Word Cloud"
                    className="w-full h-auto rounded-lg"
                  />
                ) : (
                  <p className="text-gray-500">
                    Word cloud could not be generated.
                  </p>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Features Section */}
      <section id="features" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Core Features
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              What our Natural Language Processing model can do
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {/* Feature 1: Classification */}
            <div className="text-center p-8 rounded-2xl border border-green-100 hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mx-auto mb-6">
                <ClipboardCheck className="w-6 h-6 text-green-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                Ternary Classification
              </h3>
              <p className="text-gray-600 mb-6">
                Accurately classifies tweets into three distinct sentiment
                categories: **Positive, Negative, and Neutral**.
              </p>
            </div>

            {/* Feature 2: Visualization */}
            <div className="text-center p-8 rounded-2xl border border-blue-100 hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-6">
                <BarChart3 className="w-6 h-6 text-blue-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                Confidence Scoring
              </h3>
              <p className="text-gray-600 mb-6">
                Provides the **model's probability score** for the prediction,
                allowing users to gauge the analysis reliability.
              </p>
            </div>

            {/* Feature 3: Tokenization/Preprocessing */}
            <div className="text-center p-8 rounded-2xl border border-orange-100 hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 bg-orange-100 rounded-xl flex items-center justify-center mx-auto mb-6">
                <Lightbulb className="w-6 h-6 text-orange-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                Robust Preprocessing
              </h3>
              <p className="text-gray-600 mb-6">
                Handles noise inherent in social media data (emojis, URLs,
                slang) through **lemmatization** and **tokenization**.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {/* Brand Section */}
            <div className="col-span-1 md:col-span-2">
              <div className="flex items-center space-x-3 mb-6">
                <div className="bg-gradient-to-r from-orange-500 to-green-600 p-3 rounded-lg">
                  <Brain className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold bg-gradient-to-r from-orange-400 to-green-400 bg-clip-text text-transparent">
                    TweetSense
                  </h3>
                  <p className="text-gray-400 text-sm">NLP Mini Project</p>
                </div>
              </div>

              <p className="text-gray-300 mb-6 leading-relaxed max-w-md">
                A simple Natural Language Processing application to showcase
                tweet sentiment analysis as a final year mini-project.
              </p>

              <div className="flex space-x-4">
                <a
                  href="https://twitter.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="bg-gray-800 p-3 rounded-lg hover:bg-gray-700 transition-colors"
                >
                  <Twitter className="w-5 h-5" />
                </a>
                <a
                  href="https://github.com/your-repo-link"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="bg-gray-800 p-3 rounded-lg hover:bg-gray-700 transition-colors"
                >
                  <Code className="w-5 h-5" />
                </a>
              </div>
            </div>

            {/* Links */}
            <div>
              <h4 className="text-lg font-semibold mb-4">Project Focus</h4>
              <ul className="space-y-3 text-gray-300">
                <li>Machine Learning</li>
                <li>Python & Libraries</li>
                <li>Data Preprocessing</li>
                <li>Text Visualization</li>
              </ul>
            </div>

            {/* Authors */}
            <div>
              <h4 className="text-lg font-semibold mb-4">Team/Author</h4>
              <ul className="space-y-3 text-gray-300">
                <li>[Your Name/Team Member 1]</li>
                <li>[Team Member 2]</li>
                <li>[Supervisor Name]</li>
              </ul>
            </div>
          </div>

          {/* Bottom Footer */}
          <div className="border-t border-gray-800 mt-12 pt-8">
            <div className="flex flex-col md:flex-row justify-between items-center">
              <div className="flex items-center space-x-4 mb-4 md:mb-0">
                <p className="text-gray-400 text-sm">
                  © 2025 TweetSense Analyzer. NLP Mini Project.
                </p>
                <div className="hidden md:flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-gray-400 text-xs">API Ready</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default SentimentHomePage;
