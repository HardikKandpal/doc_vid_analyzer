// Update the health check function to use the correct port
import axios from 'axios';
import { API_URL, API_ENDPOINTS } from '../config';

// Create an axios instance with the correct base URL
const api = axios.create({
  baseURL: API_URL,
  timeout: 15000, // Increased timeout for slower connections
  headers: {
    'Content-Type': 'application/json',
  }
});

// Add request interceptor for authentication
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add a function to check backend connectivity
export const checkBackendConnection = async () => {
  try {
    console.log(`Attempting to connect to backend at: ${API_ENDPOINTS.HEALTH}`);
    
    // Use a timeout to avoid long waits
    const response = await axios.get(API_ENDPOINTS.HEALTH, {
      timeout: 5000,  // 5 second timeout
    });
    
    console.log('Backend connection successful:', response.data);
    return true;
  } catch (error) {
    console.error('Backend connection failed:', error);
    
    // Log more detailed error information
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('Error response data:', error.response.data);
      console.error('Error response status:', error.response.status);
    } else if (error.request) {
      // The request was made but no response was received
      console.error('No response received from server');
    }
    
    return false;
  }
};

export default api;