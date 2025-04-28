import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  CircularProgress,
  Container,
  List,
  ListItem,
  ListItemText,
  Divider,
  Alert,
  Card,
  CardContent
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ChatIcon from '@mui/icons-material/Chat';
import VideocamIcon from '@mui/icons-material/Videocam';
import ApiService from '../services/api';

const VideoChatbotPage = () => {
  const [taskId, setTaskId] = useState('');
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [chatHistory, setChatHistory] = useState([]);

  const handleTaskIdChange = (event) => {
    setTaskId(event.target.value);
  };

  const handleQueryChange = (event) => {
    setQuery(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!taskId.trim()) {
      setError('Please enter the Task ID from your video analysis.');
      return;
    }

    if (!query.trim()) {
      setError('Please enter a question.');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      setChatHistory(prev => [
        ...prev,
        { role: 'user', content: query }
      ]);

      const response = await ApiService.legalChatbot(query, taskId);

      setChatHistory(prev => [
        ...prev,
        { role: 'assistant', content: response.answer }
      ]);

      setQuery('');
    } catch (error) {
      console.error('Error getting chatbot response:', error);
      setError('An error occurred while processing your question. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          <VideocamIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Video Q&A Assistant
        </Typography>
        
        <Typography variant="body1" paragraph>
          Ask questions about your previously analyzed <strong>videos</strong>.
          Enter the Task ID you received after video analysis.
        </Typography>

        <Paper sx={{ p: 3, mb: 4 }}>
          <TextField
            fullWidth
            label="Video Task ID"
            variant="outlined"
            value={taskId}
            onChange={handleTaskIdChange}
            placeholder="Enter the Task ID from your video analysis"
            margin="normal"
            helperText="This connects your questions to the specific video you analyzed"
          />
          
          {error && (
            <Alert severity="error" sx={{ mt: 2, mb: 2 }}>
              {error}
            </Alert>
          )}
        </Paper>

        <Card variant="outlined" sx={{ mb: 3 }}>
          <CardContent sx={{ p: 0 }}>
            <Box sx={{ height: '400px', display: 'flex', flexDirection: 'column' }}>
              <Box sx={{ p: 2, bgcolor: 'primary.main', color: 'white' }}>
                <Typography variant="h6">
                  <ChatIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Video Chat
                </Typography>
              </Box>
              
              <Box sx={{ 
                flexGrow: 1, 
                overflow: 'auto', 
                p: 2,
                bgcolor: 'background.default'
              }}>
                {chatHistory.length === 0 ? (
                  <Box sx={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    height: '100%',
                    color: 'text.secondary'
                  }}>
                    <VideocamIcon sx={{ fontSize: 60, mb: 2, opacity: 0.5 }} />
                    <Typography variant="body1">
                      No messages yet. Start by asking a question about your video.
                    </Typography>
                  </Box>
                ) : (
                  <List>
                    {chatHistory.map((message, index) => (
                      <React.Fragment key={index}>
                        <ListItem 
                          alignItems="flex-start"
                          sx={{ 
                            bgcolor: message.role === 'assistant' ? 'background.paper' : 'transparent',
                            borderRadius: 2,
                            mb: 1
                          }}
                        >
                          <ListItemText
                            primary={message.role === 'user' ? 'You' : 'Video Assistant'}
                            secondary={message.content}
                            primaryTypographyProps={{
                              fontWeight: 'bold',
                              color: message.role === 'assistant' ? 'primary.main' : 'text.primary'
                            }}
                            secondaryTypographyProps={{
                              variant: 'body1',
                              color: 'text.primary',
                              whiteSpace: 'pre-line'
                            }}
                          />
                        </ListItem>
                        {index < chatHistory.length - 1 && <Divider component="li" />}
                      </React.Fragment>
                    ))}
                  </List>
                )}
              </Box>
              
              <Divider />
              
              <Box sx={{ p: 2, bgcolor: 'background.paper' }}>
                <form onSubmit={handleSubmit}>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <TextField
                      fullWidth
                      variant="outlined"
                      placeholder="Ask a question about your video..."
                      value={query}
                      onChange={handleQueryChange}
                      disabled={isLoading || !taskId.trim()}
                      size="small"
                    />
                    <Button
                      variant="contained"
                      color="primary"
                      endIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
                      type="submit"
                      disabled={isLoading || !query.trim() || !taskId.trim()}
                    >
                      {isLoading ? 'Sending' : 'Send'}
                    </Button>
                  </Box>
                </form>
              </Box>
            </Box>
          </CardContent>
        </Card>

        <Paper sx={{ p: 3, bgcolor: 'background.paper' }}>
          <Typography variant="h6" gutterBottom>
            Example Video Questions
          </Typography>
          <List dense>
            <ListItem>
              <ListItemText primary="What legal terms were discussed in the video?" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Summarize the main points of the video." />
            </ListItem>
            <ListItem>
              <ListItemText primary="Were any contract clauses mentioned?" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Explain the risk assessment from the video." />
            </ListItem>
            <ListItem>
              <ListItemText primary="What are the key takeaways from this video?" />
            </ListItem>
          </List>
        </Paper>
      </Box>
    </Container>
  );
};

export default VideoChatbotPage;