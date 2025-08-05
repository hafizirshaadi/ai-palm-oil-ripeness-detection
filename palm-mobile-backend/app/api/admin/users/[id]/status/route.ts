import { NextRequest, NextResponse } from "next/server"
import { getServerSession } from "next-auth"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import connectDB from '@/lib/mongodb';
import User from "@/models/User"

export async function PATCH(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
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
    
    // Extract user ID from params
    const userId = params.id
    
    // Parse request body
    const { status } = await req.json()
    
    // Validate status
    if (status !== "active" && status !== "suspended") {
      return NextResponse.json(
        { message: "Invalid status value" },
        { status: 400 }
      )
    }
    
    // Connect to the database
    await connectDB()
    
    // Find the user
    const user = await User.findById(userId)
    
    if (!user) {
      return NextResponse.json(
        { message: "User not found" },
        { status: 404 }
      )
    }
    
    // Prevent admins from suspending other admins (optional security measure)
    if (user.isAdmin && status === "suspended") {
      return NextResponse.json(
        { message: "Cannot suspend admin users" },
        { status: 403 }
      )
    }
    
    // Update user status
    user.status = status
    await user.save()
    
    return NextResponse.json({
      message: `User status updated to ${status}`,
      status
    })
  } catch (error) {
    console.error("Error updating user status:", error)
    return NextResponse.json(
      { message: "Internal server error" },
      { status: 500 }
    )
  }
}
