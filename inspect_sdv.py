# inspect_sdv.py
import sdv
import sdv.multi_table

print("--- Contents of 'sdv.multi_table' ---")

# List all available names in the submodule
available_names = dir(sdv.multi_table)

# Filter for synthesizer classes, which typically end in 'Synthesizer'
synthesizers = [name for name in available_names if 'Synthesizer' in name]

if synthesizers:
    print("\nFound the following synthesizer classes:")
    for synth in synthesizers:
        print(f"- {synth}")
else:
    print("\nNo classes ending in 'Synthesizer' found directly in this submodule.")

print("\n" + "="*50 + "\n")
print("Full directory listing for debugging:")
print(available_names)