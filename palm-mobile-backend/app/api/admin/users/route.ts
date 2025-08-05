import { NextRequest, NextResponse } from "next/server"
import { getServerSession } from "next-auth"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import connectDB from '@/lib/mongodb';
import User from "@/models/User"

export async function GET(req: NextRequest) {
  try {
    // Check if the user is authenticated and is an admin
    const session = await getServerSession(authOptions)
    
    if (!session) {
      return NextResponse.json(
        { message: "Unauthorized" },
        { status: 401 }
      )
    }
    
    // @ts-ignore - Adding type for session.user
    if (!session.user.isAdmin) {
      return NextResponse.json(
        { message: "Admin access required" },
        { status: 403 }
      )
    }
    
    // Connect to the database
    await connectDB()
    
    // Get all users, sorted by creation date (newest first)
    const users = await User.find({})
      .select('name email status isAdmin createdAt')
      .sort({ createdAt: -1 })
    
    return NextResponse.json(users)
  } catch (error) {
    console.error("Error fetching users:", error)
    return NextResponse.json(
      { message: "Internal server error" },
      { status: 500 }
    )
  }
}
