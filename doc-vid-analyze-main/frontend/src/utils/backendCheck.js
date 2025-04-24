import axios from 'axios';
import { API_BASE_URL } from '../config';

/**
 * Check if the backend server is running
 * @returns {Promise<boolean>} True if backend is running, false otherwise
 */
export const checkBackendConnection = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`, { timeout: 5000 });
    return response.status === 200;
  } catch (error) {
    console.error('Backend connection error:', error);
    return false;
  }
};