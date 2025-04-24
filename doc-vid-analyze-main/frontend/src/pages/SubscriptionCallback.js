import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Box, Typography, CircularProgress } from '@mui/material';
import ApiService from '../services/api';

function SubscriptionCallback({ showNotification }) {
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState('processing');
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    const verifySubscription = async () => {
      try {
        const params = new URLSearchParams(location.search);
        const status = params.get('status');
        const subscriptionId = params.get('subscription_id');
        
        if (status === 'success' && subscriptionId) {
          // Verify the subscription with the backend
          const response = await ApiService.verifySubscription(subscriptionId);
          
          if (response.status === 'success') {
            setStatus('success');
            showNotification('Subscription activated successfully!', 'success');
            // Redirect to dashboard after 3 seconds
            setTimeout(() => navigate('/dashboard'), 3000);
          } else {
            setStatus('error');
            showNotification('Failed to verify subscription', 'error');
          }
        } else if (status === 'cancel') {
          setStatus('cancelled');
          showNotification('Subscription was cancelled', 'info');
          // Redirect to subscription page after 3 seconds
          setTimeout(() => navigate('/subscription'), 3000);
        } else {
          setStatus('error');
          showNotification('Invalid subscription callback', 'error');
        }
      } catch (error) {
        console.error('Subscription verification error:', error);
        setStatus('error');
        showNotification('Error verifying subscription', 'error');
      } finally {
        setLoading(false);
      }
    };

    verifySubscription();
  }, [location, navigate, showNotification]);

  return (
    <Box sx={{ py: 8, textAlign: 'center' }}>
      {loading ? (
        <CircularProgress />
      ) : (
        <Box>
          {status === 'success' && (
            <>
              <Typography variant="h4" gutterBottom>
                Subscription Activated!
              </Typography>
              <Typography variant="body1">
                Your subscription has been successfully activated. You will be redirected to the dashboard.
              </Typography>
            </>
          )}
          
          {status === 'cancelled' && (
            <>
              <Typography variant="h4" gutterBottom>
                Subscription Cancelled
              </Typography>
              <Typography variant="body1">
                Your subscription process was cancelled. You will be redirected to the subscription page.
              </Typography>
            </>
          )}
          
          {status === 'error' && (
            <>
              <Typography variant="h4" gutterBottom>
                Subscription Error
              </Typography>
              <Typography variant="body1">
                There was an error processing your subscription. Please try again later.
              </Typography>
            </>
          )}
        </Box>
      )}
    </Box>
  );
}

export default SubscriptionCallback;