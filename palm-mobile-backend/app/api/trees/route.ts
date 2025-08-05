import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import { Tree, TreeHistory } from '@/models/Tree';

/**
 * Get all trees
 */
export async function GET() {
  try {
    console.log('GET /api/trees - Getting all trees');
    await connectDB();
    
    const trees = await Tree.find({}).sort({ timestamp: -1 });
    console.log(`Found ${trees.length} trees`);
    
    return NextResponse.json(trees, { status: 200 });
  } catch (error) {
    console.error('Error fetching trees:', error);
    return NextResponse.json(
      { message: 'Failed to fetch trees' }, 
      { status: 500 }
    );
  }
}

/**
 * Create a new tree
 */
export async function POST(request: Request) {
  try {
    console.log('POST /api/trees - Creating new tree');
    await connectDB();
    
    const data = await request.json();
    console.log('Request data:', data);
    
    // Validate required fields
    if (!data.x || !data.y || !data.status || !data.description) {
      console.log('Missing required fields:', { x: data.x, y: data.y, status: data.status, description: data.description });
      return NextResponse.json(
        { message: 'Missing required fields' }, 
        { status: 400 }
      );
    }
    
    // Create the tree
    const newTree = await Tree.create({
      x: data.x,
      y: data.y,
      status: data.status,
      description: data.description,
      timestamp: new Date()
    });
    console.log('Tree created successfully:', newTree);
    
    // Create initial history entry
    await TreeHistory.create({
      treeId: newTree._id,
      status: newTree.status,
      description: newTree.description,
      timestamp: newTree.timestamp
    });
    
    return NextResponse.json(newTree, { status: 201 });
  } catch (error) {
    console.error('Error creating tree:', error);
    return NextResponse.json(
      { message: 'Failed to create tree' }, 
      { status: 500 }
    );
  }
}
