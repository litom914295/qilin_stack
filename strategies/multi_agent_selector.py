# Multi-Agent Stock Selector - Simplified Version
# Author: Warp AI Assistant
# Date: 2025-01

import pandas as pd
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)

@dataclass
class AgentScore:
    agent_name: str
    score: float
    confidence: float
    signals: Dict[str, Any]
    explanation: str

class FundamentalAgent:
    def score(self, df, code=None, fundamentals=None):
        if not fundamentals:
            return AgentScore("Fundamental", 50.0, 0.3, {}, "No data")
        try:
            scores, signals = {}, {}
            if "pe" in fundamentals:
                pe = fundamentals["pe"]
                if 0 < pe < 15: 
                    scores["pe"], signals["pe"] = 80, f"Low PE({pe:.1f})"
                elif pe < 30: 
                    scores["pe"], signals["pe"] = 60, f"OK PE({pe:.1f})"
                elif pe < 50: 
                    scores["pe"], signals["pe"] = 40, f"High PE({pe:.1f})"
                else: 
                    scores["pe"], signals["pe"] = 20, f"VHigh PE({pe:.1f})"
            
            if "pb" in fundamentals:
                pb = fundamentals["pb"]
                if 0 < pb < 1.5: 
                    scores["pb"], signals["pb"] = 80, f"Low PB({pb:.2f})"
                elif pb < 3: 
                    scores["pb"], signals["pb"] = 60, f"OK PB({pb:.2f})"
                else: 
                    scores["pb"], signals["pb"] = 40, f"High PB({pb:.2f})"
            
            if "roe" in fundamentals:
                roe = fundamentals["roe"] * 100
                if roe > 20: 
                    scores["roe"], signals["roe"] = 90, f"High ROE({roe:.1f}%%)"
                elif roe > 15: 
                    scores["roe"], signals["roe"] = 70, f"Good ROE({roe:.1f}%%)"
                elif roe > 10: 
                    scores["roe"], signals["roe"] = 50, f"OK ROE({roe:.1f}%%)"
                else: 
                    scores["roe"], signals["roe"] = 30, f"Low ROE({roe:.1f}%%)"
            
            total_score = np.mean(list(scores.values())) if scores else 50.0
            confidence = len(scores) / 3.0 if scores else 0.1
            explanation = ", ".join([f"{k}:{v}" for k, v in signals.items()])
            return AgentScore("Fundamental", total_score, confidence, signals, explanation)
        except Exception as e:
            logger.error(f"Fundamental error: {e}")
            return AgentScore("Fundamental", 50.0, 0.0, {}, f"Error: {e}")

class MultiAgentStockSelector:
    def __init__(self, chanlun_weight=0.7, fundamental_weight=0.3, 
                 enable_chanlun=True, enable_fundamental=True):
        self.weights = {
            "chanlun": chanlun_weight if enable_chanlun else 0, 
            "fundamental": fundamental_weight if enable_fundamental else 0
        }
        total = sum(self.weights.values())
        if total > 0: 
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        self.agents = {}
        if enable_chanlun:
            try:
                from agents.chanlun_agent import ChanLunScoringAgent
                self.agents["chanlun"] = ChanLunScoringAgent()
            except ImportError:
                logger.warning("ChanLun agent unavailable")
        if enable_fundamental:
            self.agents["fundamental"] = FundamentalAgent()
        logger.info(f"Initialized with weights: {self.weights}")
    
    def score(self, df, code=None, fundamentals=None, return_details=False):
        agent_scores = {}
        if "chanlun" in self.agents:
            agent_scores["chanlun"] = self.agents["chanlun"].score(df, code)
        if "fundamental" in self.agents:
            agent_scores["fundamental"] = self.agents["fundamental"].score(df, code, fundamentals)
        
        total_score = total_confidence = 0.0
        for name, weight in self.weights.items():
            if name in agent_scores:
                s = agent_scores[name]
                score = s.score if isinstance(s, AgentScore) else s
                conf = s.confidence if isinstance(s, AgentScore) else 1.0
                total_score += score * weight
                total_confidence += conf * weight
        
        if return_details:
            return total_score, {
                "total_score": total_score, 
                "confidence": total_confidence, 
                "agent_scores": agent_scores, 
                "weights": self.weights, 
                "grade": self._get_grade(total_score)
            }
        return total_score
    
    def batch_score(self, stock_data, fundamentals_data=None, top_n=None):
        results = []
        for code, df in stock_data.items():
            fundamentals = fundamentals_data.get(code) if fundamentals_data else None
            try:
                total_score, details = self.score(df, code, fundamentals, return_details=True)
                row = {
                    "code": code, 
                    "score": total_score, 
                    "confidence": details["confidence"], 
                    "grade": details["grade"]
                }
                for name, agent_score in details["agent_scores"].items():
                    if isinstance(agent_score, AgentScore):
                        row[f"{name}_score"] = agent_score.score
                        row[f"{name}_signal"] = agent_score.explanation
                    else:
                        row[f"{name}_score"] = agent_score
                results.append(row)
            except Exception as e:
                logger.error(f"Batch error for {code}: {e}")
        
        df_result = pd.DataFrame(results)
        if len(df_result) > 0:
            df_result = df_result.sort_values("score", ascending=False)
            if top_n: 
                df_result = df_result.head(top_n)
        return df_result
    
    def _get_grade(self, score):
        if score >= 85: return "Strong Buy"
        elif score >= 70: return "Buy"
        elif score >= 55: return "Neutral+"
        elif score >= 40: return "Neutral"
        elif score >= 25: return "Wait"
        else: return "Avoid"
