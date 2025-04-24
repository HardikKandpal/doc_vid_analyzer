import React, { useState } from 'react';
import { Box, Typography, Button, Card, CardContent, CardActions, Grid, Divider, List, ListItem, ListItemIcon, ListItemText, CircularProgress, Alert } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ApiService from '../services/api';

function Subscription({ showNotification }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubscribe = async (tier) => {
    try {
      setLoading(true);
      setError(null); // Clear any previous errors
      const response = await ApiService.createSubscription(tier);
      
      if (response.success && response.data.approval_url) {
        // Redirect to PayPal for payment
        window.location.href = response.data.approval_url;
      } else {
        setError('Failed to create subscription. Please try again.');
        showNotification('Subscription Error', 'error');
      }
    } catch (error) {
      console.error('Subscription error:', error);
      setError('An error occurred. Please try again later.');
      showNotification('Subscription Error', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ py: 6, px: 2, maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h3" component="h1" align="center" gutterBottom>
        Choose Your Subscription Plan
      </Typography>
      
      <Typography variant="h6" align="center" color="textSecondary" paragraph sx={{ mb: 6 }}>
        Select the plan that best fits your needs
      </Typography>
      
      {/* Display error message if there is one */}
      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error}
        </Alert>
      )}
      
      <Grid container spacing={4} justifyContent="center">
        {/* Free Tier */}
        <Grid item xs={12} sm={6} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <Typography variant="h4" component="h2" gutterBottom>
                Free
              </Typography>
              <Typography variant="h3" component="div" color="primary" gutterBottom>
                ₹0
              </Typography>
              <Typography variant="subtitle1" color="textSecondary" gutterBottom>
                per month
              </Typography>
              <Divider sx={{ my: 2 }} />
              <List>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary="Basic document analysis" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary="Limited API calls" />
                </ListItem>
              </List>
            </CardContent>
            <CardActions>
              <Button fullWidth variant="outlined" disabled>
                Current Plan
              </Button>
            </CardActions>
          </Card>
        </Grid>
        
        {/* Standard Tier */}
        <Grid item xs={12} sm={6} md={4}>
          <Card sx={{ 
            height: '100%', 
            display: 'flex', 
            flexDirection: 'column',
            border: '2px solid #6200EA',
            transform: 'scale(1.05)',
            position: 'relative'
          }}>
            <Box sx={{ 
              position: 'absolute', 
              top: 0, 
              right: 0, 
              backgroundColor: 'primary.main', 
              color: 'white',
              px: 2,
              py: 0.5,
              borderBottomLeftRadius: 8
            }}>
              Popular
            </Box>
            <CardContent sx={{ flexGrow: 1 }}>
              <Typography variant="h4" component="h2" gutterBottom>
                Standard
              </Typography>
              <Typography variant="h3" component="div" color="primary" gutterBottom>
                ₹799
              </Typography>
              <Typography variant="subtitle1" color="textSecondary" gutterBottom>
                per month
              </Typography>
              <Divider sx={{ my: 2 }} />
              <List>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary="Advanced document analysis" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary="Video analysis" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary="Priority support" />
                </ListItem>
              </List>
            </CardContent>
            <CardActions>
              <Button 
                fullWidth 
                variant="contained" 
                onClick={() => handleSubscribe('standard_tier')}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} color="inherit" /> : 'Subscribe Now'}
              </Button>
            </CardActions>
          </Card>
        </Grid>
        
        {/* Premium Tier */}
        <Grid item xs={12} sm={6} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <Typography variant="h4" component="h2" gutterBottom>
                Premium
              </Typography>
              <Typography variant="h3" component="div" color="primary" gutterBottom>
                ₹1499
              </Typography>
              <Typography variant="subtitle1" color="textSecondary" gutterBottom>
                per month
              </Typography>
              <Divider sx={{ my: 2 }} />
              <List>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary="Everything in Standard" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary="Unlimited API calls" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary="Advanced analytics" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary="24/7 support" />
                </ListItem>
              </List>
            </CardContent>
            <CardActions>
              <Button 
                fullWidth 
                variant="contained" 
                color="secondary"
                onClick={() => handleSubscribe('premium_tier')}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} color="inherit" /> : 'Subscribe Now'}
              </Button>
            </CardActions>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Subscription;