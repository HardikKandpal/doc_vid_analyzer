import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Box, Typography, CircularProgress, Paper, Button } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import ApiService from '../services/api';

function PayPalReturn({ showNotification }) {
  const [status, setStatus] = useState('processing');
  const location = useLocation();
  const navigate = useNavigate();
  
  // Define verifyPayment inside the useEffect or use useCallback
  useEffect(() => {
    const verifyPayment = async (subscriptionId) => {
      try {
        const response = await ApiService.verifySubscription(subscriptionId);
        
        if (response.success) {
          setStatus('success');
          showNotification('Subscription activated successfully!', 'success');
        } else {
          setStatus('error');
          showNotification('Failed to verify subscription', 'error');
        }
      } catch (error) {
        console.error('Verification error:', error);
        setStatus('error');
        showNotification('Error verifying subscription', 'error');
      }
    };

    const queryParams = new URLSearchParams(location.search);
    const paymentStatus = queryParams.get('status');
    const subscriptionId = queryParams.get('subscription_id');
    
    if (paymentStatus === 'success' && subscriptionId) {
      verifyPayment(subscriptionId);
    } else if (paymentStatus === 'cancel') {
      setStatus('cancelled');
      showNotification('Subscription was cancelled', 'info');
    } else {
      setStatus('error');
      showNotification('Invalid payment response', 'error');
    }
  }, [location, showNotification, navigate]);
  
  return (
    <Box sx={{ py: 8, px: 2, maxWidth: 600, mx: 'auto', textAlign: 'center' }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
        {status === 'processing' && (
          <>
            <CircularProgress size={60} thickness={4} sx={{ mb: 3 }} />
            <Typography variant="h5" gutterBottom>
              Processing Your Subscription
            </Typography>
            <Typography color="textSecondary">
              Please wait while we verify your payment...
            </Typography>
          </>
        )}
        
        {status === 'success' && (
          <>
            <CheckCircleIcon color="success" sx={{ fontSize: 60, mb: 3 }} />
            <Typography variant="h5" gutterBottom>
              Subscription Activated!
            </Typography>
            <Typography color="textSecondary" paragraph>
              Your subscription has been successfully activated. You now have access to all the features of your plan.
            </Typography>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={() => navigate('/')}
              sx={{ mt: 2 }}
            >
              Go to Dashboard
            </Button>
          </>
        )}
        
        {status === 'cancelled' && (
          <>
            <ErrorIcon color="warning" sx={{ fontSize: 60, mb: 3 }} />
            <Typography variant="h5" gutterBottom>
              Subscription Cancelled
            </Typography>
            <Typography color="textSecondary" paragraph>
              You cancelled the subscription process. No payment was made.
            </Typography>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={() => navigate('/subscription')}
              sx={{ mt: 2 }}
            >
              Back to Subscription Plans
            </Button>
          </>
        )}
        
        {status === 'error' && (
          <>
            <ErrorIcon color="error" sx={{ fontSize: 60, mb: 3 }} />
            <Typography variant="h5" gutterBottom>
              Something Went Wrong
            </Typography>
            <Typography color="textSecondary" paragraph>
              We couldn't verify your subscription. Please contact support if you believe this is an error.
            </Typography>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={() => navigate('/subscription')}
              sx={{ mt: 2 }}
            >
              Try Again
            </Button>
          </>
        )}
      </Paper>
    </Box>
  );
}

export default PayPalReturn;