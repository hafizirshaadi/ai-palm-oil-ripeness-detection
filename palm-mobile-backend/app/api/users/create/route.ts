import { NextRequest, NextResponse } from "next/server"
import connectDB from '@/lib/mongodb';
import User from "@/models/User"

export async function POST(req: NextRequest) {
  try {
    const { name, email, password, status, isAdmin } = await req.json()
    
    // Validate required fields
    if (!name || !email || !password) {
      return NextResponse.json(
        { message: "Please provide all required fields" },
        { status: 400 }
      )
    }
    
    // Connect to database
    await connectDB()
    
    // Check if email already exists
    const existingUser = await User.findOne({ email })
    
    if (existingUser) {
      return NextResponse.json(
        { message: "Email already in use" },
        { status: 400 }
      )
    }
    
    // Create new user
    const user = await User.create({
      name,
      email,
      password,
      status: status || "active", // Default to active if not specified
      isAdmin: isAdmin || false, // Default to false if not specified
    })
    
    // Return success response without password
    return NextResponse.json({
      message: "User created successfully",
      user: {
        _id: user._id,
        name: user.name,
        email: user.email,
        status: user.status,
        isAdmin: user.isAdmin,
      }
    }, { status: 201 })
    
  } catch (error) {
    console.error("Error creating user:", error)
    return NextResponse.json(
      { message: "Error creating user" },
      { status: 500 }
    )
  }
}
