import * as React from 'react';

interface ResetPasswordEmailProps {
  resetUrl: string;
  userName: string;
}

export const ResetPasswordEmail: React.FC<Readonly<ResetPasswordEmailProps>> = ({
  resetUrl,
  userName,
}) => (
  <div style={container}>
    <div style={header}>
      <h1 style={heading}>PALM Mobile Password Reset</h1>
    </div>
    
    <div style={body}>
      <p style={paragraph}>Hello {userName},</p>
      <p style={paragraph}>
        We received a request to reset your password for your PALM Mobile account. If you didn't make this request, you can safely ignore this email.
      </p>
      <p style={paragraph}>
        To reset your password, use the following code:
      </p>
      
      <div style={codeContainer}>
        <p style={codeStyle}>{resetUrl.split('/').pop()}</p>
      </div>
      
      <p style={paragraph}>
        This code will expire in 1 hour for security reasons.
      </p>
      
      <p style={paragraph}>
        If you have any questions, please don't hesitate to contact our support team.
      </p>
      
      <p style={paragraph}>
        Thank you,<br />
        The PALM Mobile Team
      </p>
    </div>
    
    <div style={footer}>
      <p style={footerText}>
        © {new Date().getFullYear()} PALM Mobile. All rights reserved.
      </p>
    </div>
  </div>
);

// Styles
const container = {
  fontFamily: 'Arial, sans-serif',
  maxWidth: '600px',
  margin: '0 auto',
  backgroundColor: '#f9f9f9',
  borderRadius: '8px',
  overflow: 'hidden',
};

const header = {
  backgroundColor: '#2e8b57',
  padding: '20px',
  textAlign: 'center' as const,
};

const heading = {
  color: 'white',
  margin: '0',
  fontSize: '24px',
};

const body = {
  padding: '20px',
  backgroundColor: 'white',
};

const paragraph = {
  fontSize: '16px',
  lineHeight: '1.5',
  color: '#333',
  margin: '16px 0',
};

const codeContainer = {
  margin: '24px 0',
  padding: '12px',
  textAlign: 'center' as const,
  backgroundColor: '#f5f5f5',
  borderRadius: '4px',
  border: '1px solid #e0e0e0',
};

const codeStyle = {
  fontSize: '24px',
  fontWeight: 'bold' as const,
  margin: '0',
  color: '#2e8b57',
  letterSpacing: '2px',
};

const footer = {
  backgroundColor: '#f4f4f4',
  padding: '15px',
  textAlign: 'center' as const,
};

const footerText = {
  fontSize: '12px',
  color: '#666',
  margin: '0',
};
