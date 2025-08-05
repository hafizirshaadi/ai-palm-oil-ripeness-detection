import { NextRequest, NextResponse } from 'next/server';
import connectMongoDB from '@/lib/mongodb';
import EducationResource from '@/models/EducationResource';

export async function PUT(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params;
    const body = await request.json();
    const { userId } = body;

    if (!userId) {
      return NextResponse.json(
        { error: 'User ID is required' },
        { status: 400 }
      );
    }

    await connectMongoDB();
    
    const resource = await EducationResource.findById(id);
    
    if (!resource) {
      return NextResponse.json(
        { error: 'Education resource not found' },
        { status: 404 }
      );
    }

    // Toggle like
    const isLiked = resource.likes.includes(userId);
    
    if (isLiked) {
      // Remove like
      await EducationResource.findByIdAndUpdate(
        id,
        { $pull: { likes: userId } },
        { new: true }
      );
    } else {
      // Add like
      await EducationResource.findByIdAndUpdate(
        id,
        { $addToSet: { likes: userId } },
        { new: true }
      );
    }

    const updatedResource = await EducationResource.findById(id);
    return NextResponse.json(updatedResource);
  } catch (error) {
    console.error('Error toggling like:', error);
    return NextResponse.json(
      { error: 'Failed to toggle like' },
      { status: 500 }
    );
  }
}
