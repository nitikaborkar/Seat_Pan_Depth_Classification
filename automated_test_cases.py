import pytest
import os
import json
from seat_depth_analysis import process_seat_depth_analysis

class TestSeatDepthAnalysis:
    """Test cases for seat depth analysis"""
    
    def setup_method(self):
        """Setup test environment"""
        self.eye_to_ear_cm = 7.0
        self.sam_checkpoint = "sam_vit_b_01ec64.pth"
        self.sample_images_dir = "sample_images"
    
    def test_optimal_classification(self):
        """Test that optimal samples are classified correctly"""
        optimal_dir = os.path.join(self.sample_images_dir, "optimal")
        if not os.path.exists(optimal_dir):
            pytest.skip("Optimal sample images not found")
        
        for image_file in os.listdir(optimal_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_path = os.path.join(optimal_dir, image_file)
                
                try:
                    output_json, _, _, _ = process_seat_depth_analysis(
                        image_path, self.eye_to_ear_cm, self.sam_checkpoint
                    )
                    
                    # Test classification
                    assert output_json['classification']['category'] == "Optimal", \
                        f"Expected 'Optimal' for {image_file}, got {output_json['classification']['category']}"
                    
                    # Test clearance range
                    clearance = output_json['measurements']['knee_clearance_cm']
                    assert 2.0 <= clearance <= 6.0, \
                        f"Optimal clearance should be 2-6cm, got {clearance:.2f}cm for {image_file}"
                    
                    print(f"✅ {image_file}: {output_json['classification']['category']} - {clearance:.2f}cm")
                    
                except Exception as e:
                    pytest.fail(f"Failed to process {image_file}: {str(e)}")
    
    def test_too_deep_classification(self):
        """Test that too deep samples are classified correctly"""
        too_deep_dir = os.path.join(self.sample_images_dir, "too_deep")
        if not os.path.exists(too_deep_dir):
            pytest.skip("Too deep sample images not found")
        
        for image_file in os.listdir(too_deep_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_path = os.path.join(too_deep_dir, image_file)
                
                try:
                    output_json, _, _, _ = process_seat_depth_analysis(
                        image_path, self.eye_to_ear_cm, self.sam_checkpoint
                    )
                    
                    # Test classification
                    assert output_json['classification']['category'] == "Too Deep", \
                        f"Expected 'Too Deep' for {image_file}, got {output_json['classification']['category']}"
                    
                    # Test that clearance is less than 2cm OR knee is behind seat
                    clearance = output_json['measurements']['knee_clearance_cm']
                    knee_behind = output_json['classification']['knee_behind_seat']
                    
                    assert clearance < 2.0 or knee_behind, \
                        f"Too Deep should have <2cm clearance or knee behind seat, got {clearance:.2f}cm, behind: {knee_behind}"
                    
                    print(f"✅ {image_file}: {output_json['classification']['category']} - {clearance:.2f}cm")
                    
                except Exception as e:
                    pytest.fail(f"Failed to process {image_file}: {str(e)}")
    
    def test_too_short_classification(self):
        """Test that too short samples are classified correctly"""
        too_short_dir = os.path.join(self.sample_images_dir, "too_short")
        if not os.path.exists(too_short_dir):
            pytest.skip("Too short sample images not found")
        
        for image_file in os.listdir(too_short_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_path = os.path.join(too_short_dir, image_file)
                
                try:
                    output_json, _, _, _ = process_seat_depth_analysis(
                        image_path, self.eye_to_ear_cm, self.sam_checkpoint
                    )
                    
                    # Test classification
                    assert output_json['classification']['category'] == "Too Short", \
                        f"Expected 'Too Short' for {image_file}, got {output_json['classification']['category']}"
                    
                    # Test clearance range
                    clearance = output_json['measurements']['knee_clearance_cm']
                    assert clearance > 6.0, \
                        f"Too Short clearance should be >6cm, got {clearance:.2f}cm for {image_file}"
                    
                    print(f"✅ {image_file}: {output_json['classification']['category']} - {clearance:.2f}cm")
                    
                except Exception as e:
                    pytest.fail(f"Failed to process {image_file}: {str(e)}")
    
    def test_edge_cases(self):
        """Test handling of edge cases"""
        # Test invalid image path
        with pytest.raises(ValueError, match="Could not load image"):
            process_seat_depth_analysis("nonexistent_image.jpg", self.eye_to_ear_cm, self.sam_checkpoint)
    
    def test_json_output_structure(self):
        """Test that JSON output has required structure"""
        # Use any available sample image
        test_image = None
        for category in ["optimal", "too_deep", "too_short"]:
            category_dir = os.path.join(self.sample_images_dir, category)
            if os.path.exists(category_dir):
                for image_file in os.listdir(category_dir):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        test_image = os.path.join(category_dir, image_file)
                        break
                if test_image:
                    break
        
        if not test_image:
            pytest.skip("No test images available")
        
        output_json, _, _, _ = process_seat_depth_analysis(
            test_image, self.eye_to_ear_cm, self.sam_checkpoint
        )
        
        # Test required top-level keys
        required_keys = [
            'frame_id', 'timestamp', 'pose_detection', 'chair_detection',
            'measurements', 'classification', 'debug_info', 'warnings', 'processing_time_ms'
        ]
        
        for key in required_keys:
            assert key in output_json, f"Missing required key: {key}"
        
        # Test classification structure
        assert 'category' in output_json['classification']
        assert 'reasoning' in output_json['classification']
        assert output_json['classification']['category'] in ["Optimal", "Too Deep", "Too Short"]
        
        # Test measurements structure
        measurement_keys = [
            'knee_clearance_cm', 'knee_clearance_px', 'pixels_per_cm',
            'seat_front_position', 'back_of_knee_position'
        ]
        
        for key in measurement_keys:
            assert key in output_json['measurements'], f"Missing measurement key: {key}"
        
        print("✅ JSON output structure is valid")

def run_validation_report():
    """Generate a validation report for all sample images"""
    print("\n" + "="*60)
    print("SEAT DEPTH ANALYSIS VALIDATION REPORT")
    print("="*60)
    
    eye_to_ear_cm = 7.0
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    sample_images_dir = "sample_images"
    
    categories = ["optimal", "too_deep", "too_short"]
    
    for category in categories:
        print(f"\n📁 {category.upper()} SAMPLES:")
        print("-" * 40)
        
        category_dir = os.path.join(sample_images_dir, category)
        if not os.path.exists(category_dir):
            print(f"❌ Directory not found: {category_dir}")
            continue
        
        image_files = [f for f in os.listdir(category_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        if not image_files:
            print(f"❌ No images found in {category_dir}")
            continue
        
        correct_classifications = 0
        total_images = len(image_files)
        
        for image_file in image_files:
            image_path = os.path.join(category_dir, image_file)
            
            try:
                output_json, _, _, _ = process_seat_depth_analysis(
                    image_path, eye_to_ear_cm, sam_checkpoint
                )
                
                predicted_category = output_json['classification']['category']
                clearance = output_json['measurements']['knee_clearance_cm']
                expected_category = category.replace("_", " ").title()
                
                is_correct = predicted_category == expected_category
                if is_correct:
                    correct_classifications += 1
                
                status = "✅" if is_correct else "❌"
                print(f"{status} {image_file}: {predicted_category} ({clearance:.2f}cm)")
                
            except Exception as e:
                print(f"❌ {image_file}: ERROR - {str(e)}")
        
        accuracy = (correct_classifications / total_images * 100) if total_images > 0 else 0
        print(f"\n📊 {category.upper()} Accuracy: {correct_classifications}/{total_images} ({accuracy:.1f}%)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Run validation report
    run_validation_report()
    
    # Run pytest
    print("\nRunning automated tests...")
    pytest.main([__file__, "-v"])