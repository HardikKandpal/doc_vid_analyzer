// API configuration
export const API_BASE_URL = process.env.REACT_APP_API_URL || "https://huggingface.co/spaces/hardik8588/doc_analyzer/";
// Other configuration options
export const APP_NAME = "Legal Document & Video Analyzer";
export const DEFAULT_TIMEOUT = 30000; // 30 seconds

// API endpoints
export const API_ENDPOINTS = {
  REGISTER: `${API_BASE_URL}/register`,
  LOGIN: `${API_BASE_URL}/token`,
  USER_INFO: `${API_BASE_URL}/users/me`,
  
  // Health check endpoint
  HEALTH: `${API_BASE_URL}/health`,
  
  // Subscription endpoints
  CREATE_SUBSCRIPTION: `${API_BASE_URL}/create_subscription`,
  GET_USER_SUBSCRIPTION: `${API_BASE_URL}/users/me/subscription`,
  SUBSCRIBE: `${API_BASE_URL}/subscribe`,
  VERIFY_SUBSCRIPTION: `${API_BASE_URL}/subscription/verify`,
  
  // Analysis endpoints
  ANALYZE_DOCUMENT: `${API_BASE_URL}/analyze_legal_document`,
  ANALYZE_VIDEO: `${API_BASE_URL}/analyze_legal_video`,
  ANALYZE_AUDIO: `${API_BASE_URL}/analyze_legal_audio`,
  LEGAL_CHATBOT: `${API_BASE_URL}/legal_chatbot`
};

// Subscription tiers
export const SUBSCRIPTION_TIERS = {
  FREE_TIER: 'free_tier',
  STANDARD_TIER: 'standard_tier',
  PREMIUM_TIER: 'premium_tier'
};

// Other configuration
export const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
