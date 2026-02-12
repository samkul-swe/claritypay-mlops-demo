"""
MongoDB integration for logging predictions
"""
import os
from datetime import datetime
from pymongo import MongoClient
from typing import Dict, Any, Optional


class PredictionLogger:
    """Logs predictions to MongoDB for monitoring and analytics"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.client = None
        self.db = None
        
        # Get MongoDB URI from environment variable
        mongodb_uri = os.getenv("MONGODB_URI")
        if mongodb_uri:
            try:
                self.client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
                # Test connection
                self.client.admin.command('ping')
                self.db = self.client.credit_scoring
                print("✅ MongoDB connected")
            except Exception as e:
                print(f"⚠️  MongoDB connection failed: {e}")
                self.client = None
                self.db = None
        else:
            print("ℹ️  No MONGODB_URI found - predictions won't be logged")
    
    def log_prediction(self, 
                      application: Dict[str, Any], 
                      prediction: Dict[str, Any]) -> Optional[str]:
        """
        Log prediction to MongoDB
        
        Returns: Document ID if successful, None otherwise
        """
        if not self.db:
            return None
        
        try:
            document = {
                "timestamp": datetime.utcnow(),
                "application": application,
                "prediction": prediction,
                "model_version": "1.0.0"
            }
            
            result = self.db.predictions.insert_one(document)
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"⚠️  Failed to log prediction: {e}")
            return None
    
    def get_recent_predictions(self, limit: int = 10) -> list:
        """Get recent predictions for monitoring"""
        if not self.db:
            return []
        
        try:
            cursor = self.db.predictions.find(
                {}, 
                {"_id": 0}  # Exclude MongoDB internal ID
            ).sort("timestamp", -1).limit(limit)
            
            # Convert datetime to string for JSON serialization
            predictions = []
            for doc in cursor:
                doc["timestamp"] = doc["timestamp"].isoformat()
                predictions.append(doc)
            
            return predictions
        except Exception as e:
            print(f"⚠️  Failed to get predictions: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        if not self.db:
            return {"connected": False, "message": "MongoDB not connected"}
        
        try:
            total = self.db.predictions.count_documents({})
            
            if total == 0:
                return {
                    "connected": True,
                    "total_predictions": 0,
                    "approval_rate": 0,
                    "message": "No predictions logged yet"
                }
            
            # Approval rate
            approved = self.db.predictions.count_documents({
                "prediction.approval_recommendation": "APPROVED"
            })
            
            # Average credit score
            pipeline = [
                {"$group": {
                    "_id": None,
                    "avg_credit_score": {"$avg": "$prediction.credit_score"}
                }}
            ]
            avg_result = list(self.db.predictions.aggregate(pipeline))
            avg_score = avg_result[0]["avg_credit_score"] if avg_result else 0
            
            return {
                "connected": True,
                "total_predictions": total,
                "approval_rate": round(approved / total, 3) if total > 0 else 0,
                "average_credit_score": round(avg_score, 0)
            }
        except Exception as e:
            return {"connected": True, "error": str(e)}
