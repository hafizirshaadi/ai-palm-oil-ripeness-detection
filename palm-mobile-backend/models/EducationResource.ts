import mongoose from 'mongoose';

const EducationResourceSchema = new mongoose.Schema(
  {
    userId: {
      type: String,
      required: true,
    },
    userName: {
      type: String,
      required: true,
    },
    content: {
      type: String,
      required: true,
    },
    mediaUrl: {
      type: String,
    },
    youtubeUrl: {
      type: String,
    },
    likes: {
      type: [String],
      default: [],
    },
  },
  { timestamps: true }
);

export default mongoose.models.EducationResource ||
  mongoose.model('EducationResource', EducationResourceSchema);
