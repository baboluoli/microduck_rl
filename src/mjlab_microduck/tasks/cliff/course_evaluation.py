"""Independent scalar classifier for pre-reset evidence, including both feet."""
import math


def classify(samples, *, height, dt=.02):
    contact_x = None
    reasons = set()
    one_metre = False
    success = False
    end = None
    for sample in samples:
        end = sample
        # Each foot preserves found, position, and force even when not found.
        for foot in sample['feet']:
            p = foot['position']
            if (foot['found'] and all(math.isfinite(v) for v in p)
                    and .55 <= p[0] <= 4.05 and abs(p[1]) <= 1.35
                    and abs(p[2]+height) <= .008 and contact_x is None):
                contact_x = sample['x']
        if not sample['finite']: reasons.add('nonfinite')
        if sample['head'] > 15 or sample['trunk'] > 20: reasons.add('impact')
        if contact_x is not None and sample['cos_tilt'] <= math.cos(math.pi/4): reasons.add('tilt')
        if sample['failure']: reasons.add('physical_failure')
        if sample['x'] < -.75 or sample['x'] > 3.8 or abs(sample['y']) > 1.15: reasons.add('boundary')
        if contact_x is not None and sample['x']-contact_x >= 1 and not reasons:
            one_metre = True
        if contact_x is not None and sample['x'] >= 3.55 and not reasons:
            success = True
            break
    return dict(success=success, one_metre=one_metre, contact=contact_x is not None,
                contact_x=contact_x, late_contact=contact_x is not None and 3.55-contact_x < 1,
                distance_after_contact=None if contact_x is None or end is None else end['x']-contact_x,
                duration=0 if end is None else end['age']*dt, finish_x=3.55,
                events=sorted(reasons), outcome='success' if success else
                'boundary' if 'boundary' in reasons else 'failure' if reasons else 'timeout')
