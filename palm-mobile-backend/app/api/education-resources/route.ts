import { NextRequest, NextResponse } from 'next/server';
import connectMongoDB from '@/lib/mongodb';
import EducationResource from '@/models/EducationResource';

export async function GET(request: NextRequest) {
  try {
    await connectMongoDB();
    const resources = await EducationResource.find({}).sort({ createdAt: -1 });
    return NextResponse.json(resources);
  } catch (error) {
    console.error('Error fetching education resources:', error);
    return NextResponse.json(
      { error: 'Failed to fetch education resources' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { userId, userName, content, mediaUrl, youtubeUrl } = body;

    if (!userId || !userName || !content) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    await connectMongoDB();
    
    const newResource = await EducationResource.create({
      userId,
      userName,
      content,
      mediaUrl,
      youtubeUrl,
      likes: [],
    });

    return NextResponse.json(newResource, { status: 201 });
  } catch (error) {
    console.error('Error creating education resource:', error);
    return NextResponse.json(
      { error: 'Failed to create education resource' },
      { status: 500 }
    );
  }
}
