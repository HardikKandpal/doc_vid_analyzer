import axios from 'axios';
import { API_BASE_URL, API_ENDPOINTS } from '../config';

// Create axios instance with the correct base URL
const api = axios.create({
  baseURL: API_BASE_URL
});

// Add request interceptor to include token in requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Authentication functions
// Register a new user
const register = async (email, password) => {
  try {
    // Use the correct endpoint with the base URL
    const response = await api.post(API_ENDPOINTS.REGISTER, { email, password });
    
    if (response.data && response.data.access_token) {
      // Store the token in localStorage
      localStorage.setItem('access_token', response.data.access_token);
      return { success: true, data: response.data };
    } else {
      return { success: false, error: 'Registration failed' };
    }
  } catch (error) {
    console.error('Registration error:', error);
    return { 
      success: false, 
      error: error.response?.data?.detail || 'Registration failed' 
    };
  }
};

const login = async (email, password) => {
  try {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    
    const response = await axios.post(API_ENDPOINTS.LOGIN, formData);
    
    // Remove token_type from destructuring since it's not used
    const { access_token } = response.data;
    
    // Store token in localStorage
    localStorage.setItem('access_token', access_token);
    
    // Get user info
    const userInfo = await getUserInfo();
    localStorage.setItem('user_info', JSON.stringify(userInfo.data));
    
    return { success: true, data: userInfo.data };
  } catch (error) {
    return { 
      success: false, 
      error: error.response?.data?.detail || 'Login failed' 
    };
  }
};

const logout = () => {
  localStorage.removeItem('access_token');
  localStorage.removeItem('user_info');
  window.location.href = '/login';
};

const getUserInfo = async () => {
  try {
    const response = await api.get(API_ENDPOINTS.USER_INFO);
    return { success: true, data: response.data };
  } catch (error) {
    return { 
      success: false, 
      error: error.response?.data?.detail || 'Failed to get user info' 
    };
  }
};

const isAuthenticated = () => {
  return !!localStorage.getItem('access_token');
};

const getCurrentUser = () => {
  const userInfo = localStorage.getItem('user_info');
  return userInfo ? JSON.parse(userInfo) : null;
};

// Create a named export object
const authService = {
  register,
  login,
  logout,
  getUserInfo,
  isAuthenticated,
  getCurrentUser,
  api
};

export default authService;