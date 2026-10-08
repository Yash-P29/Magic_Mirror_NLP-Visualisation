import unittest
import json
from unittest.mock import MagicMock

from integration.runtime.multimodal_runtime import MultimodalRuntime
from integration.multimodal_fusion.schemas import ActionType, AnchorType

# --- Helper Mocks ---

def create_nlp_json(text: str, intent: str, entities: list = None, corefs: list = None, rels: list = None, confidence: float = 0.9):
    data = {
        "timestamp": 12345.0,
        "is_partial": False,
        "text_segment": text,
        "speaker_activity": [{"intent": intent, "confidence": confidence}],
        "semantic_graph": {
            "entities": entities or [],
            "relationships": rels or []
        },
        "coreference_resolutions": corefs or [],
        "explanation_state": {
            "current_topic": "Test",
            "active_concept": "Test",
            "recent_entities": []
        }
    }
    return json.dumps(data)

def create_vision_scene(hand_available=False, table_available=False):
    scene = MagicMock()
    scene.timestamp = 12345.0
    scene.frame_id = 1
    scene.activity = MagicMock()
    scene.activity.left_hand_gesture.value = "none"
    scene.activity.right_hand_gesture.value = "none"
    scene.activity.motion = "none"
    scene.relationships = []
    
    if hand_available:
        scene.visual_opportunity = MagicMock()
        scene.visual_opportunity.available = True
        scene.visual_opportunity.type.value = "on_hand"
        scene.visual_opportunity.bbox = (0.4, 0.4, 0.6, 0.6)
        scene.visual_opportunity.center = (0.5, 0.5)
    elif table_available:
        scene.visual_opportunity = MagicMock()
        scene.visual_opportunity.available = True
        scene.visual_opportunity.type.value = "pointing_target" # simulating table via pointing
        scene.visual_opportunity.bbox = (0.2, 0.2, 0.8, 0.8)
        scene.visual_opportunity.center = (0.5, 0.5)
    else:
        scene.visual_opportunity = None
        
    return scene


class TestIntegrationLayer(unittest.TestCase):
    
    def setUp(self):
        self.runtime = MultimodalRuntime()

    def test_1_anecdotal_speech_none(self):
        text = "I saw a dog yesterday."
        nlp_json = create_nlp_json(text, "casual", entities=[{"id": "dog_1", "name": "dog", "class": "animal", "status": "new_explicit"}])
        plan = self.runtime.process_nlp_update(nlp_json)
        self.assertEqual(plan.action, ActionType.NONE)

    def test_2_imagine_dog_create(self):
        text = "Imagine a dog."
        nlp_json = create_nlp_json(text, "hypothetical", entities=[{"id": "dog_1", "name": "dog", "class": "animal", "status": "new_explicit"}])
        plan = self.runtime.process_nlp_update(nlp_json)
        self.assertEqual(plan.action, ActionType.CREATE)
        self.assertEqual(plan.visual_id, "dog_1")

    def test_3_imagine_dog_on_hand(self):
        text = "Imagine a dog sitting on my hand."
        nlp_json = create_nlp_json(text, "hypothetical", entities=[{"id": "dog_1", "name": "dog", "class": "animal", "status": "new_explicit"}], rels=[{"subject": "dog_1", "predicate": "sitting_on", "object": "hand_1", "type": "spatial"}])
        
        vision_scene = create_vision_scene(hand_available=True)
        self.runtime.process_vision_update(vision_scene)
        
        plan = self.runtime.process_nlp_update(nlp_json)
        self.assertEqual(plan.action, ActionType.CREATE)
        self.assertTrue(plan.spatial.anchored)
        self.assertEqual(plan.spatial.anchor_type, AnchorType.HAND)

    def test_4_make_it_stand_update(self):
        # Create dog first
        self.runtime.process_nlp_update(create_nlp_json("Imagine a dog.", "hypothetical", entities=[{"id": "dog_1", "name": "dog", "class": "animal", "status": "new_explicit"}]))
        
        text = "Make it stand."
        nlp_json = create_nlp_json(text, "instructing", corefs=[{"text_reference": "it", "resolved_entity": "dog_1"}])
        plan = self.runtime.process_nlp_update(nlp_json)
        self.assertEqual(plan.action, ActionType.ANIMATE)
        self.assertEqual(plan.visual_id, "dog_1")
        self.assertEqual(plan.animation.state, "standing")

    def test_5_make_it_run_animate(self):
        self.runtime.process_nlp_update(create_nlp_json("Imagine a dog.", "hypothetical", entities=[{"id": "dog_1", "name": "dog", "class": "animal", "status": "new_explicit"}]))
        
        text = "Make it run."
        nlp_json = create_nlp_json(text, "instructing", corefs=[{"text_reference": "it", "resolved_entity": "dog_1"}])
        plan = self.runtime.process_nlp_update(nlp_json)
        self.assertEqual(plan.action, ActionType.ANIMATE)
        self.assertEqual(plan.visual_id, "dog_1")
        self.assertEqual(plan.animation.state, "running")

    def test_6_move_onto_table(self):
        self.runtime.process_nlp_update(create_nlp_json("Imagine a dog.", "hypothetical", entities=[{"id": "dog_1", "name": "dog", "class": "animal", "status": "new_explicit"}]))
        self.runtime.process_vision_update(create_vision_scene(table_available=True))
        
        text = "Move it onto the table."
        nlp_json = create_nlp_json(text, "instructing", corefs=[{"text_reference": "it", "resolved_entity": "dog_1"}])
        plan = self.runtime.process_nlp_update(nlp_json)
        self.assertEqual(plan.action, ActionType.MOVE)
        self.assertTrue(plan.spatial.anchored)

    def test_7_remove_it(self):
        self.runtime.process_nlp_update(create_nlp_json("Imagine a dog.", "hypothetical", entities=[{"id": "dog_1", "name": "dog", "class": "animal", "status": "new_explicit"}]))
        
        text = "Now remove it."
        nlp_json = create_nlp_json(text, "instructing", corefs=[{"text_reference": "it", "resolved_entity": "dog_1"}])
        plan = self.runtime.process_nlp_update(nlp_json)
        self.assertEqual(plan.action, ActionType.REMOVE)
        self.assertEqual(plan.visual_id, "dog_1")

    def test_8_10_earth_sun_sequence(self):
        # 8
        p1 = self.runtime.process_nlp_update(create_nlp_json("Imagine Earth.", "hypothetical", entities=[{"id": "earth_1", "name": "Earth", "class": "planet", "status": "new_explicit"}]))
        self.assertEqual(p1.action, ActionType.CREATE)
        
        # 9
        p2 = self.runtime.process_nlp_update(create_nlp_json("Make it rotate.", "instructing", corefs=[{"text_reference": "it", "resolved_entity": "earth_1"}]))
        self.assertEqual(p2.action, ActionType.ANIMATE)
        
        # 10
        p3 = self.runtime.process_nlp_update(create_nlp_json("Put the Sun beside it.", "instructing", 
                                                             entities=[{"id": "sun_1", "name": "Sun", "class": "star", "status": "new_explicit"}],
                                                             corefs=[{"text_reference": "it", "resolved_entity": "earth_1"}]))
        self.assertIn(p3.action, [ActionType.CREATE, ActionType.UPDATE])
        self.assertEqual(p3.visual_id, "sun_1")

    def test_14_unresolvable_reference(self):
        text = "Put it there."
        nlp_json = create_nlp_json(text, "instructing", corefs=[]) # no resolved entity
        plan = self.runtime.process_nlp_update(nlp_json)
        # Should be NONE because we have no entity
        self.assertEqual(plan.action, ActionType.NONE)

    def test_15_water_molecule(self):
        text = "The water molecule contains two hydrogen atoms and one oxygen atom."
        nlp_json = create_nlp_json(text, "explaining", entities=[{"id": "mol_1", "name": "molecule", "class": "chemistry", "status": "new_explicit"}])
        plan = self.runtime.process_nlp_update(nlp_json)
        self.assertEqual(plan.action, ActionType.CREATE)
        self.assertEqual(plan.representation.type.value, "3d")

if __name__ == '__main__':
    unittest.main()
