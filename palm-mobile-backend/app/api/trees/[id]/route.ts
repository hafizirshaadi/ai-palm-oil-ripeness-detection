import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import { Tree, TreeHistory } from '@/models/Tree';

interface Params {
  params: {
    id: string;
  }
}

/**
 * Get a specific tree by ID
 */
export async function GET(request: Request, { params }: Params) {
  try {
    await connectDB();
    
    const tree = await Tree.findById(params.id);
    
    if (!tree) {
      return NextResponse.json(
        { message: 'Tree not found' },
        { status: 404 }
      );
    }
    
    return NextResponse.json(tree, { status: 200 });
  } catch (error) {
    console.error('Error fetching tree:', error);
    return NextResponse.json(
      { message: 'Failed to fetch tree' }, 
      { status: 500 }
    );
  }
}

/**
 * Update a tree by ID
 */
export async function PUT(request: Request, { params }: Params) {
  try {
    await connectDB();
    
    const data = await request.json();
    
    // Find the tree
    const tree = await Tree.findById(params.id);
    
    if (!tree) {
      return NextResponse.json(
        { message: 'Tree not found' },
        { status: 404 }
      );
    }
    
    // Update tree
    const updatedTree = await Tree.findByIdAndUpdate(
      params.id,
      {
        status: data.status,
        description: data.description,
        timestamp: new Date()
      },
      { new: true }
    );
    
    // Create history entry for the update
    await TreeHistory.create({
      treeId: params.id,
      status: data.status,
      description: data.description,
      timestamp: new Date()
    });
    
    return NextResponse.json(updatedTree, { status: 200 });
  } catch (error) {
    console.error('Error updating tree:', error);
    return NextResponse.json(
      { message: 'Failed to update tree' }, 
      { status: 500 }
    );
  }
}

/**
 * Delete a tree by ID
 */
export async function DELETE(request: Request, { params }: Params) {
  try {
    await connectDB();
    
    // Find the tree
    const tree = await Tree.findById(params.id);
    
    if (!tree) {
      return NextResponse.json(
        { message: 'Tree not found' },
        { status: 404 }
      );
    }
    
    // Delete the tree
    await Tree.findByIdAndDelete(params.id);
    
    // Delete all history entries for this tree
    await TreeHistory.deleteMany({ treeId: params.id });
    
    return NextResponse.json(
      { message: 'Tree deleted successfully' }, 
      { status: 200 }
    );
  } catch (error) {
    console.error('Error deleting tree:', error);
    return NextResponse.json(
      { message: 'Failed to delete tree' }, 
      { status: 500 }
    );
  }
}
