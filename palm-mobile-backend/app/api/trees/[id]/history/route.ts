import { NextResponse } from "next/server";
import connectDB from "@/lib/mongodb";
import { TreeHistory } from "@/models/Tree";

interface Params {
  params: {
    id: string;
  };
}

/**
 * Get history for a specific tree
 */
export async function GET(request: Request, { params }: Params) {
  try {
    await connectDB();

    const history = await TreeHistory.find({ treeId: params.id }).sort({
      timestamp: -1,
    });

    return NextResponse.json(history, { status: 200 });
  } catch (error) {
    console.error("Error fetching tree history:", error);
    return NextResponse.json(
      { message: "Failed to fetch tree history" },
      { status: 500 }
    );
  }
}
