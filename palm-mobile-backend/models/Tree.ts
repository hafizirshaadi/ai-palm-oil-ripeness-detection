import mongoose from 'mongoose';

// Define the schema for tree history entries
const TreeHistorySchema = new mongoose.Schema({
  treeId: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'Tree', 
    required: true 
  },
  status: { 
    type: String, 
    enum: ['healthy', 'sick'], 
    required: true 
  },
  description: { 
    type: String, 
    required: true 
  },
  timestamp: { 
    type: Date, 
    default: Date.now 
  }
});

// Define the schema for trees
const TreeSchema = new mongoose.Schema({
  x: { 
    type: Number, 
    required: true 
  },
  y: { 
    type: Number, 
    required: true 
  },
  status: { 
    type: String, 
    enum: ['healthy', 'sick'], 
    required: true 
  },
  description: { 
    type: String, 
    required: true 
  },
  timestamp: { 
    type: Date, 
    default: Date.now 
  }
});

// Create models if they don't exist yet
export const Tree = mongoose.models.Tree || mongoose.model('Tree', TreeSchema);
export const TreeHistory = mongoose.models.TreeHistory || mongoose.model('TreeHistory', TreeHistorySchema);
