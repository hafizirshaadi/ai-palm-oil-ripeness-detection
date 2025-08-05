import { NextRequest, NextResponse } from 'next/server';
import connectMongoDB from '@/lib/mongodb';
import EducationResource from '@/models/EducationResource';

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params;
    await connectMongoDB();
    const resource = await EducationResource.findById(id);

    if (!resource) {
      return NextResponse.json(
        { error: 'Education resource not found' },
        { status: 404 }
      );
    }

    return NextResponse.json(resource);
  } catch (error) {
    console.error('Error fetching education resource:', error);
    return NextResponse.json(
      { error: 'Failed to fetch education resource' },
      { status: 500 }
    );
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params;
    const body = await request.json();
    const { content, mediaUrl, youtubeUrl } = body;

    await connectMongoDB();
    
    const resource = await EducationResource.findById(id);
    
    if (!resource) {
      return NextResponse.json(
        { error: 'Education resource not found' },
        { status: 404 }
      );
    }

    const updatedResource = await EducationResource.findByIdAndUpdate(
      id,
      { content, mediaUrl, youtubeUrl },
      { new: true }
    );

    return NextResponse.json(updatedResource);
  } catch (error) {
    console.error('Error updating education resource:', error);
    return NextResponse.json(
      { error: 'Failed to update education resource' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params;
    await connectMongoDB();
    
    const resource = await EducationResource.findById(id);
    
    if (!resource) {
      return NextResponse.json(
        { error: 'Education resource not found' },
        { status: 404 }
      );
    }

    await EducationResource.findByIdAndDelete(id);

    return NextResponse.json({ message: 'Education resource deleted successfully' });
  } catch (error) {
    console.error('Error deleting education resource:', error);
    return NextResponse.json(
      { error: 'Failed to delete education resource' },
      { status: 500 }
    );
  }
}
