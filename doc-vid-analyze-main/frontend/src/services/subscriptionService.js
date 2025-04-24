import authService from './authService';
import { API_ENDPOINTS } from '../config';

// Get user subscription
const getUserSubscription = async () => {
  try {
    const response = await authService.api.get(API_ENDPOINTS.GET_USER_SUBSCRIPTION);
    return { success: true, data: response.data };
  } catch (error) {
    return { 
      success: false, 
      error: error.response?.data?.detail || 'Failed to get subscription info' 
    };
  }
};

// Create subscription - Updated to use the correct endpoint
const createSubscription = async (tier) => {
  try {
    const response = await authService.api.post(API_ENDPOINTS.CREATE_SUBSCRIPTION, { tier });
    return { success: true, data: response.data };
  } catch (error) {
    console.error('Subscription error:', error);
    return { 
      success: false, 
      error: error.response?.data?.detail || 'Failed to create subscription' 
    };
  }
};

// Verify subscription - Used after PayPal redirect
const verifySubscription = async (subscriptionId) => {
  try {
    const response = await authService.api.get(
      `${API_ENDPOINTS.VERIFY_SUBSCRIPTION}/${subscriptionId}`
    );
    
    return { success: true, data: response.data };
  } catch (error) {
    return { 
      success: false, 
      error: error.response?.data?.detail || 'Failed to verify subscription' 
    };
  }
};

// Cancel subscription
const cancelSubscription = async () => {
  try {
    const response = await authService.api.post(API_ENDPOINTS.CANCEL_SUBSCRIPTION);
    return { success: true, data: response.data };
  } catch (error) {
    return { 
      success: false, 
      error: error.response?.data?.detail || 'Failed to cancel subscription' 
    };
  }
};

// Downgrade to free tier
const downgradeToFreeTier = async () => {
  try {
    const response = await authService.api.post(`${API_ENDPOINTS.CANCEL_SUBSCRIPTION}/downgrade`);
    return { success: true, data: response.data };
  } catch (error) {
    return { 
      success: false, 
      error: error.response?.data?.detail || 'Failed to downgrade subscription' 
    };
  }
};

// Create a named export object
const subscriptionService = {
  getUserSubscription,
  createSubscription,
  verifySubscription,
  cancelSubscription,
  downgradeToFreeTier
};

export default subscriptionService;