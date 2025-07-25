import axios from 'axios';
import { API_BASE_URL, API_ENDPOINTS } from '../config';


// Create an axios instance with the base URL
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor to include the auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add the checkHealth function
const checkHealth = async () => {
  try {
    const response = await api.get(API_ENDPOINTS.HEALTH);
    return response.status === 200;
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};

// Create the ApiService object with all methods
const ApiService = {
  checkHealth,
  
  // Create a subscription
  createSubscription: async (tier) => {
    try {
      const response = await api.post(`/subscription/create`, { tier });
      console.log("Subscription API response:", response.data);
      return response.data; // Return the data directly as it comes from the server
    } catch (error) {
      console.error('Create subscription error:', error);
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Failed to create subscription'
      };
    }
  },

  // Verify a subscription after payment
  verifySubscription: async (subscriptionId) => {
    try {
      const response = await api.post(`/subscription/verify`, {
        subscription_id: subscriptionId
      });
      return response.data;
    } catch (error) {
      console.error('Error verifying subscription:', error);
      throw error;
    }
  },  // Added comma here
  
  // Analyze a legal document
  analyzeDocument: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // Use the correct endpoint from your backend
      const response = await api.post('/analyze_legal_document', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error analyzing document:', error);
      
      // Handle subscription-related errors specifically
      if (error.response) {
        if (error.response.status === 403) {
          return {
            success: false,
            error: error.response.data.detail || 'This feature requires a subscription upgrade.',
            requiresUpgrade: true
          };
        } else if (error.response.status === 500) {
          // Check if the error message contains subscription information
          const errorDetail = error.response.data?.detail || '';
          if (errorDetail.includes('subscription') || errorDetail.includes('tier')) {
            return {
              success: false,
              error: errorDetail,
              requiresUpgrade: true
            };
          }
        }
      }
      
      // Generic error handling
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to analyze document'
      };
    }
  },
  
  // Analyze a legal video
  analyzeVideo: async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await api.post('/analyze_legal_video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error analyzing video:', error);

      // Handle subscription-related errors specifically
      if (error.response) {
        if (error.response.status === 403) {
          return {
            success: false,
            error: error.response.data.detail || 'This feature requires a subscription upgrade.',
            requiresUpgrade: true
          };
        } else if (error.response.status === 500) {
          const errorDetail = error.response.data?.detail || '';
          if (errorDetail.includes('subscription') || errorDetail.includes('tier')) {
            return {
              success: false,
              error: errorDetail,
              requiresUpgrade: true
            };
          }
        }
      }

      // Generic error handling
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to analyze video'
      };
    }
  },
  
  // ADD THIS FUNCTION:
  legalChatbot: async (question, taskId) => {
    try {
      const formData = new FormData();
      formData.append('question', question);

      const response = await api.post(`/legal_chatbot/${taskId}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error getting chatbot response:', error);
      throw error;
    }
  },
  
  // Visualization endpoints (return image URLs for <img> tags)
  getRiskChartUrl: () => `${API_BASE_URL}/download_risk_chart`,
  getRiskPieChartUrl: () => `${API_BASE_URL}/download_risk_pie_chart`,
  getRiskRadarChartUrl: () => `${API_BASE_URL}/download_risk_radar_chart`,
  getRiskTrendChartUrl: () => `${API_BASE_URL}/download_risk_trend_chart`,
  getInteractiveRiskChartHtml: async () => {
    try {
      const response = await api.get('/interactive_risk_chart');
      return response.data; // This is HTML
    } catch (error) {
      console.error('Error fetching interactive chart:', error);
      return '<div>Error loading chart</div>';
    }
  }
};

export default ApiService;
