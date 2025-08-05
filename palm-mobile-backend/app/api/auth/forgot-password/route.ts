import { NextRequest, NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import crypto from 'crypto';
import { Resend } from 'resend';
import * as React from 'react';
import { ResetPasswordEmail } from '@/components/ResetPasswordEmail';

// Initialize Resend with API key
const resend = new Resend(process.env.RESEND_API_KEY);

export async function POST(request: NextRequest) {
  try {
    const { email } = await request.json();
    
    if (!email) {
      return NextResponse.json(
        { message: 'Email is required' },
        { status: 400 }
      );
    }
    
    await connectDB();
    
    // Find the user by email
    const user = await User.findOne({ email });
    
    // Even if the user doesn't exist, we return success to prevent email enumeration attacks
    if (!user) {
      return NextResponse.json(
        { message: 'If an account with this email exists, a password reset code has been sent.' },
        { status: 200 }
      );
    }
    
    // Generate a secure random token (6 characters for easier entry on mobile)
    const resetToken = crypto.randomBytes(3).toString('hex');
    
    // Hash the token for security (don't store plain tokens in the DB)
    const hashedToken = crypto
      .createHash('sha256')
      .update(resetToken)
      .digest('hex');
    
    // Set the token and expiration (1 hour from now)
    user.resetPasswordToken = hashedToken;
    user.resetPasswordExpires = new Date(Date.now() + 3600000); // 1 hour
    await user.save();
    
    try {
      // Send email with reset code
      const emailResult = await resend.emails.send({
        from: 'PALM Mobile <noreply@resend.dev>', // Update with your verified domain
        to: [email],
        subject: 'Reset your PALM Mobile password',
        react: React.createElement(ResetPasswordEmail, { 
          resetUrl: resetToken, // Just send the token as we only need the code
          userName: user.name || 'User'
        })
      });
      
      if (emailResult.error) {
        console.error('Email sending error:', emailResult.error);
      } else {
        console.log('Email sent successfully:', emailResult.data);
      }
    } catch (emailError) {
      console.error('Email sending failed:', emailError);
      // Continue with response even if email fails
    }
    
    return NextResponse.json(
      { message: 'If an account with this email exists, a password reset link has been sent.' },
      { status: 200 }
    );
  } catch (error) {
    console.error('Password reset request error:', error);
    return NextResponse.json(
      { message: 'Something went wrong' },
      { status: 500 }
    );
  }
}
